import torch


def apply_interleaved_mrope(
    frequencies: torch.Tensor,
    mrope_section: tuple[int, int, int],
) -> torch.Tensor:
    """Apply interleaved MRoPE to frequencies.

    Args:
        frequencies: (3, B, S, D/2)
        mrope_section: (t, h, w)
    """
    # (3, B, S, D/2) -> (B, S, D/2)
    frequencies_t = frequencies[0].clone()
    # round robin THWTHW...
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        frequencies_t[..., offset:length:3] = frequencies[dim, ..., offset:length:3]
    return frequencies_t


def build_mrope_frequencies(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    mrope_sections: tuple[int, int, int],
) -> torch.Tensor:
    """Build frequencies for interleaved MRoPE.

    Args:
        position_ids: (B, S)
        head_dim: int
        theta: float
        mrope_sections: (t, h, w)
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    assert sum(mrope_sections) == head_dim // 2, "Sum of mrope sections must be equal to head dimension"
 
    device = position_ids.device
    # inv_freq[i] = theta ** (-(2i)/head_dim), i = 0..D/2-1
    # (D/2,)
    theta_numerator = torch.arange(0, head_dim, 2)
    inverse_frequencies = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # (3, B, S, 1) * (D/2,) → (3, B, S, D/2)
    frequencies_3d = position_ids[..., None].float() * inverse_frequencies
    # (3, B, S, D/2) -> (B, S, D/2)
    frequencies = apply_interleaved_mrope(frequencies_3d, mrope_sections)
    frequencies_complex = torch.polar(torch.ones_like(frequencies), frequencies)
    return frequencies_complex


def apply_rotary_embeddings(
    x: torch.Tensor,
    frequencies_complex: torch.Tensor,
    device: str | None = None,
) -> torch.Tensor:
    """Rotate Q/K with interleaved-MRoPE cis.

    Args:
        x: (B, S, H, head_dim)
        frequencies_complex: (B, S, head_dim // 2)
    """
    # (B, S, H, D) -> (B, S, H, D/2) 
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (B, S, D/2) -> (B, S, 1, D/2) 
    frequencies_complex = frequencies_complex.unsqueeze(2)

    x_rotated = x_complex * frequencies_complex
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    out = x_out.type_as(x).to(device)
    return out.to(device)