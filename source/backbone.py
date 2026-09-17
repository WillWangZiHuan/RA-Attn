"""Spatiotemporal Transformer backbone with reliability-aware attention."""

import torch
import torch.nn as nn

from .attention import TransformerBlock


class SpatiotemporalBackbone(nn.Module):
    """Add positions and a class token, then return all spatial/temporal tokens."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        heads: int,
        num_patches: int,
        max_T: int,
        attn_alpha: float = 0.5,
        gate_min: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.num_patches = num_patches
        self.max_T = max_T

        self.pos_spatial = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_temporal = nn.Parameter(torch.zeros(1, max_T, 1, embed_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, heads, alpha=attn_alpha, gate_min=gate_min,
                    use_value_gating=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_temporal, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, tokens: torch.Tensor, r_patch: torch.Tensor) -> torch.Tensor:
        """Map (B, T, L, D) tokens and (B, T, L, 1) support to updated tokens."""
        if tokens.ndim != 4:
            raise ValueError("tokens must be (B,T,L,D)")
        bsz, t_steps, n_patches, dim = tokens.shape
        if n_patches != self.num_patches or dim != self.embed_dim:
            raise ValueError("Token geometry does not match the backbone")
        if not 1 <= t_steps <= self.max_T:
            raise ValueError(f"T must lie in [1,{self.max_T}]")
        if tuple(r_patch.shape) != (bsz, t_steps, n_patches, 1):
            raise ValueError("r_patch must be (B,T,L,1) and align with tokens")

        x = tokens + self.pos_spatial
        x = x + self.pos_temporal[:, :t_steps, :, :]
        x = x.reshape(bsz, t_steps * n_patches, dim)
        r_flat = r_patch.reshape(bsz, t_steps * n_patches, 1).to(
            device=tokens.device, dtype=tokens.dtype
        )

        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        r_cls = torch.ones((bsz, 1, 1), device=tokens.device, dtype=tokens.dtype)
        r_all = torch.cat([r_cls, r_flat], dim=1)

        for block in self.blocks:
            x = block(x, r_all, collect_influence=False)

        x = self.norm(x)
        return x[:, 1:, :].reshape(bsz, t_steps, n_patches, dim)
