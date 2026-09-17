"""Complete optical-only RA-Attn architecture and forward computation."""

import math

import torch
import torch.nn as nn

from .backbone import SpatiotemporalBackbone
from .encoders import LandsatPatchEncoder
from .reliability import ReliabilityBuilder
from .token_gate import ReliabilityTokenGate


class NDVIMapHead(nn.Module):
    """Decode final-year spatial tokens into a single-channel delta-NDVI map."""

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, tokens_last_t: torch.Tensor, grid_hw) -> torch.Tensor:
        bsz, n_tokens, _ = tokens_last_t.shape
        height_patches, width_patches = grid_hw
        if height_patches * width_patches != n_tokens:
            raise ValueError("Token count does not match the output patch grid")
        p = self.patch_size
        x = self.proj(tokens_last_t)
        x = x.view(bsz, height_patches, width_patches, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(bsz, 1, height_patches * p, width_patches * p)


class RAAttn(nn.Module):
    """Forecast delta NDVI from 12 annual optical images and retained counts.

    optical: (B, 12, 7, 128, 128), prepared Landsat surface reflectance.
    retained: (B, 12, 1, 128, 128), aligned QA-retained observation counts.
    output: (B, 1, 128, 128), predicted next-year NDVI minus last-input-year NDVI.

    The architecture has 8x8 patches, width 256, six blocks and eight heads.
    A temporal embedding capacity of 16 is retained from the experimental
    model; the forecasting interface uses 12 chronological input slots.
    """

    prediction_kind = "delta"

    def __init__(self, alpha: float = 0.5, tau_f: float = 0.1, tau_v: float = 0.3):
        super().__init__()
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (tau_f, tau_v)):
            raise ValueError("Gate floors must be finite and lie in [0,1]")

        # Preserve the experimental model's module construction order.
        self.rel_builder = ReliabilityBuilder(patch_size=8)
        self.enc_ms = LandsatPatchEncoder(in_chans=7, embed_dim=256, patch_size=8)
        self.token_gate = ReliabilityTokenGate(gate_min=tau_f)
        self.backbone = SpatiotemporalBackbone(
            embed_dim=256,
            depth=6,
            heads=8,
            num_patches=256,
            max_T=16,
            attn_alpha=alpha,
            gate_min=tau_v,
        )
        self.head = NDVIMapHead(embed_dim=256, patch_size=8)

    def forward(self, optical: torch.Tensor, retained: torch.Tensor) -> torch.Tensor:
        if optical.ndim != 5 or tuple(optical.shape[1:]) != (12, 7, 128, 128):
            raise ValueError("optical must be (B,12,7,128,128)")
        if tuple(retained.shape) != (optical.shape[0], 12, 1, 128, 128):
            raise ValueError("retained must be (B,12,1,128,128) and align with optical")

        _, r_patch = self.rel_builder(retained)
        optical_tokens = self.enc_ms(optical)
        gated_tokens = self.token_gate(optical_tokens, r_patch)
        tokens = self.backbone(gated_tokens, r_patch)
        return self.head(tokens[:, -1, :, :], (16, 16))
