import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from PIL import Image
from datasets import load_dataset

from ..genlip.config import Config, DataConfig
from .image_utils import count_patches, resize_for_patch_budget


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CaptionDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        tokenizer,
        split: str | None = None,
        *,
        patch_size: int = 16,
    ):
        split = split or config.train_split
        self.dataset = load_dataset(
            config.dataset_name,
            cache_dir=config.cache_dir,
            split=split,
        )
        self.tokenizer = tokenizer
        self.config = config
        self.patch_size = patch_size
        self.caption_column = config.caption_column
        self.max_text_length = config.max_text_length

        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.config.native_aspect_ratio:
            image = resize_for_patch_budget(
                image,
                patch_size=self.patch_size,
                min_patches=self.config.min_patches,
                max_patches=self.config.max_patches,
            )
        else:
            image = image.resize((self.config.image_size, self.config.image_size), Image.BICUBIC)

        x = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x

    def _encode_caption(self, caption: str) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors=None,
        )
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        pixel_values = self._process_image(row["image"])
        text = self._encode_caption(row[self.caption_column])
        _, height, width = pixel_values.shape
        num_vision_tokens = count_patches(width, height, self.patch_size)

        return {
            "pixel_values": pixel_values,
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "labels": text["labels"],
            "num_vision_tokens": num_vision_tokens,
        }


class PackedCaptionDataset(CaptionDataset):
    """Greedy patch-n-pack: concatenate samples until max_packing_length."""

    def __getitem__(self, idx: int) -> dict:
        segments: list[dict] = []
        total_len = 0
        dataset_len = len(self.dataset)

        for offset in range(dataset_len):
            sample_idx = (idx + offset) % dataset_len
            sample = super().__getitem__(sample_idx)
            seg_len = sample["num_vision_tokens"] + sample["input_ids"].numel()
            if segments and total_len + seg_len > self.config.max_packing_length:
                break
            segments.append(sample)
            total_len += seg_len
            if total_len >= self.config.max_packing_length:
                break

        if not segments:
            return super().__getitem__(idx)

        pixel_values_list = [seg["pixel_values"] for seg in segments]
        input_ids = torch.cat([seg["input_ids"] for seg in segments], dim=0)
        attention_mask = torch.cat([seg["attention_mask"] for seg in segments], dim=0)
        labels = torch.cat([seg["labels"] for seg in segments], dim=0)
        segment_vision_lengths = [seg["num_vision_tokens"] for seg in segments]
        segment_text_lengths = [seg["input_ids"].numel() for seg in segments]

        return {
            "pixel_values_list": pixel_values_list,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "segment_vision_lengths": segment_vision_lengths,
            "segment_text_lengths": segment_text_lengths,
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    if "pixel_values_list" in batch[0]:
        if len(batch) != 1:
            raise ValueError("packed batches require data.batch_size=1")
        return batch[0]

    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

    max_len = max(item["input_ids"].numel() for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = item["attention_mask"]
        labels[i, :length] = item["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataloader(
    config: Config,
    tokenizer,
    split: str | None = None,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    data_config = config.data
    split = split or data_config.train_split
    patch_size = config.model.patch_size

    if config.stage == 2:
        dataset: Dataset = PackedCaptionDataset(data_config, tokenizer, split=split, patch_size=patch_size)
    else:
        dataset = CaptionDataset(data_config, tokenizer, split=split, patch_size=patch_size)

    is_train = split == data_config.train_split
    distributed = (
        rank is not None
        and world_size is not None
        and world_size > 1
    )

    sampler = None
    shuffle = is_train and not distributed
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=data_config.drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last,
    )
