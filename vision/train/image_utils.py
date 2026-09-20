from __future__ import annotations

import math

from PIL import Image


def count_patches(width: int, height: int, patch_size: int) -> int:
    pw = max(1, width // patch_size)
    ph = max(1, height // patch_size)
    return pw * ph


def resize_for_patch_budget(
    image: Image.Image,
    *,
    patch_size: int = 16,
    min_patches: int = 16,
    max_patches: int = 1024,
) -> Image.Image:
    """Resize image at native aspect ratio so patch count stays in [min_patches, max_patches]."""
    width, height = image.size
    if width < patch_size or height < patch_size:
        scale = patch_size / min(width, height)
        width = max(patch_size, int(round(width * scale)))
        height = max(patch_size, int(round(height * scale)))

    n = count_patches(width, height, patch_size)
    if n > max_patches:
        scale = math.sqrt(n / max_patches)
        width = int(width / scale)
        height = int(height / scale)
    elif n < min_patches:
        scale = math.sqrt(min_patches / max(n, 1))
        width = int(width * scale)
        height = int(height * scale)

    width = max(patch_size, (width // patch_size) * patch_size)
    height = max(patch_size, (height // patch_size) * patch_size)

    n = count_patches(width, height, patch_size)
    while n > max_patches and width > patch_size and height > patch_size:
        if width >= height:
            width -= patch_size
        else:
            height -= patch_size
        n = count_patches(width, height, patch_size)

    while n < min_patches:
        if width <= height:
            width += patch_size
        else:
            height += patch_size
        n = count_patches(width, height, patch_size)

    return image.resize((width, height), Image.BICUBIC)
