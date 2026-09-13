"""Supporting Landsat patch encoder, provided for source reading.

The complete forecasting model and experiment workflows are not included.
"""

import torch
import torch.nn as nn


class LandsatPatchEncoder(nn.Module):
    """Encode each image in a Landsat sequence into spatial patch tokens."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x_ms: torch.Tensor) -> torch.Tensor:
        """Map (B,T,C,H,W) images to (B,T,L,D) patch tokens."""
        if x_ms.ndim != 5:
            raise ValueError(f"x_ms 期望 (B,T,C,H,W)，但拿到 {tuple(x_ms.shape)}")

        B, T, C, H, W = x_ms.shape
        xf = x_ms.reshape(B * T, C, H, W)
        feat = self.patch_embed(xf)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(B, T, tokens.shape[1], tokens.shape[2])
        return tokens
