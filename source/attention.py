"""Selected RA-Attn Transformer block, released for source reading.

The complete forecasting backbone, model assembly, and prediction head are
not included. This file provides no training or inference entry point.
Support is an observation-count proxy, not calibrated predictive uncertainty.
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LN block with a centered log-support key bias and a value gate.

    The generic constructor default is alpha=1.0; the manuscript configuration
    sets alpha=0.5. The attention value-gate floor is 0.3.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        alpha: float = 1.0,
        eps: float = 1e-6,
        gate_min: float = 0.3,
        use_value_gating: bool = True,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'dim({dim}) must be divisible by heads({heads})')
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** (-0.5)
        self.alpha = alpha
        self.eps = eps
        self.gate_min = gate_min
        self.use_value_gating = use_value_gating
        self.last_influence = None

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        r_all: torch.Tensor,
        collect_influence: bool = False,
    ) -> torch.Tensor:
        """Apply the block to tokens x (B, N, D) and aligned support (B, N, 1).

        The caller supplies finite support values in [0, 1]. If a class token is
        present, the caller supplies its support as 1 in the corresponding slot.
        When requested, last_influence stores mean incoming attention per token.
        """
        if (
            r_all.ndim != 3
            or r_all.shape[0] != x.shape[0]
            or r_all.shape[1] != x.shape[1]
            or r_all.shape[2] != 1
        ):
            raise ValueError(
                f'r_all must be (B,N,1), got {tuple(r_all.shape)} for x={tuple(x.shape)}'
            )

        bsz, n_tokens, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(bsz, n_tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # Content logits: (B, heads, queries, keys).
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Center across the full sequence; broadcast the bias along the key axis.
        r_key = r_all.squeeze(-1).to(dtype=h.dtype, device=h.device)
        log_r = torch.log(self.eps + r_key)
        log_r_centered = log_r - log_r.mean(dim=1, keepdim=True)
        key_bias = self.alpha * log_r_centered
        attn_logits = attn_logits + key_bias[:, None, None, :]

        # Scale the transmitted value of each key token.
        if self.use_value_gating:
            gate_v = self.gate_min + (1.0 - self.gate_min) * r_key
            gate_v = gate_v.to(dtype=v.dtype, device=v.device)
            v = v * gate_v[:, None, :, None]

        attn = torch.softmax(attn_logits, dim=-1)
        if collect_influence:
            # Average incoming attention over heads and query positions.
            self.last_influence = attn.mean(dim=(1, 2)).detach()
        else:
            self.last_influence = None
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, self.dim)
        attn_out = self.proj(attn_out)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
