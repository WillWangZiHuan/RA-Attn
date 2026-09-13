"""Selected core RA-Attn block for source reading.

This block applies a reliability-dependent key bias and value gate.
It is extracted from the first-revision source family retained in the
current Landsat-only working revision. Model assembly, data preparation,
training and inference workflows are not part of this partial release.
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with reliability-aware attention bias.
    Bias is added on key axis: logits += alpha * log(eps + r_key).
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
            raise ValueError(f"dim({dim}) must be divisible by heads({heads})")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
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

    def forward(self, x: torch.Tensor, r_all: torch.Tensor, collect_influence: bool = False) -> torch.Tensor:
        """
        x: (B,N,D)
        r_all: (B,N,1), reliability aligned with token order.
        """
        if r_all.ndim != 3 or r_all.shape[0] != x.shape[0] or r_all.shape[1] != x.shape[1] or r_all.shape[2] != 1:
            raise ValueError(f"r_all must be (B,N,1), got {tuple(r_all.shape)} for x={tuple(x.shape)}")

        bsz, n_tokens, _ = x.shape
        h = self.norm1(x)

        qkv = self.qkv(h).reshape(bsz, n_tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,hd)
        # Scaled query-key scores.
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,N,N)


        r_key = r_all.squeeze(-1).to(dtype=h.dtype, device=h.device)  # (B,N)
        log_r = torch.log(self.eps + r_key)  # (B,N)
        log_r_centered = log_r - log_r.mean(dim=1, keepdim=True)
        key_bias = self.alpha * log_r_centered  # (B,N)
        attn_logits = attn_logits + key_bias[:, None, None, :]

        # Reliability controls the contribution carried by each value token.
        if self.use_value_gating:
            gate_v = self.gate_min + (1.0 - self.gate_min) * r_key  # (B,N)
            gate_v = gate_v.to(dtype=v.dtype, device=v.device)
            v = v * gate_v[:, None, :, None]  # (B,H,N,hd)

        attn = torch.softmax(attn_logits, dim=-1)
        if collect_influence:
            # incoming influence: I_j = mean_{head,query} attn[...,query,j]
            self.last_influence = attn.mean(dim=(1, 2)).detach()  # (B,N)
        else:
            self.last_influence = None
        attn_out = torch.matmul(attn, v)  # (B,H,N,hd)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, self.dim)
        attn_out = self.proj(attn_out)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

