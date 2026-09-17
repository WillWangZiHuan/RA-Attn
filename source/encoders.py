"""Convolutional patch embedding for a Landsat image sequence."""

import torch
import torch.nn as nn


class LandsatPatchEncoder(nn.Module):
    """Map (B, T, C, H, W) optical inputs to (B, T, L, D) tokens."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x_ms: torch.Tensor) -> torch.Tensor:
        if x_ms.ndim != 5:
            raise ValueError(f"x_ms must be (B,T,C,H,W), got {tuple(x_ms.shape)}")
        bsz, t_steps, channels, height, width = x_ms.shape
        images = x_ms.reshape(bsz * t_steps, channels, height, width)
        features = self.patch_embed(images)
        tokens = features.flatten(2).transpose(1, 2)
        return tokens.reshape(bsz, t_steps, tokens.shape[1], tokens.shape[2])
