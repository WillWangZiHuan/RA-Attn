"""Convert QA-retained observation counts to token-aligned relative support."""

from typing import Tuple

import torch
import torch.nn.functional as F


class ReliabilityBuilder:
    """Normalize each sample/year spatially, then average-pool to token size."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, retained: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pixel support and token support from (B, T, 1, H, W) counts.

        Counts must be finite and nonnegative. The normalization denominator
        is max(1, the spatial maximum) for each sample and year independently.
        An all-zero count map therefore has zero support.
        """
        if retained.ndim != 5 or retained.shape[2] != 1:
            raise ValueError("retained must be (B,T,1,H,W)")
        height, width = retained.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Spatial dimensions must be divisible by patch_size")

        r_pixel = retained / retained.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        # Retain the experiment implementation's floating-point operation order.
        # This complement represents support deficit, not cloud probability.
        support_deficit = 1.0 - r_pixel
        r_pixel = (1.0 - support_deficit).clamp(0.0, 1.0)

        bsz, t_steps, _, height, width = r_pixel.shape
        pixels = r_pixel.reshape(bsz * t_steps, 1, height, width)
        pooled = F.avg_pool2d(
            pixels, kernel_size=self.patch_size, stride=self.patch_size
        )
        r_patch = pooled.flatten(2).transpose(1, 2)
        r_patch = r_patch.reshape(bsz, t_steps, r_patch.shape[1], 1)
        return r_pixel, r_patch
