"""Supporting conversion and pooling of an already normalized quality input.

The caller supplies quality deficit in [0,1]. This module does not read data,
normalize raw observation counts, or decode QA bitfields.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


class ReliabilityBuilder:
    """Convert quality deficit to pixel support and mean patch support."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return support shaped (B,T,1,H,W) and (B,T,L,1), respectively.

        cloud: finite (B,T,1,H,W) values in [0,1]; larger means lower support.
        Spatial dimensions must be divisible by patch_size.
        """
        if cloud.ndim != 5:
            raise ValueError(f"cloud 期望 (B,T,1,H,W)，但拿到 {tuple(cloud.shape)}")

        r_pixel = (1.0 - cloud).clamp(0.0, 1.0)

        B, T, _, H, W = r_pixel.shape
        rf = r_pixel.reshape(B * T, 1, H, W)

        r_patch_map = F.avg_pool2d(rf, kernel_size=self.patch_size, stride=self.patch_size)
        r_patch = r_patch_map.flatten(2).transpose(1, 2)
        r_patch = r_patch.reshape(B, T, r_patch.shape[1], 1)

        return r_pixel, r_patch
