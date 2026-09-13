"""Selected v35 attention block for source reading (2026-09-13 snapshot).

Extracted from the verified v35 implementation. Full model assembly, support
construction, training and inference workflows are not included.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityAttentionBlock(nn.Module):
    """Pre-LN attention with layer-local complementary Key routing and Value gating."""

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        attention_alpha: float,
        value_gate_floor: float,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim and heads must be positive, with dim divisible by heads")
        if not 0.0 <= value_gate_floor <= 1.0:
            raise ValueError("value_gate_floor must lie in [0, 1]")

        self.heads = heads
        self.head_dim = dim // heads
        self.attention_alpha = float(attention_alpha)
        self.value_gate_floor = float(value_gate_floor)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        # Each block and attention head owns a direct reliability-logit map.
        # Zeros consume no RNG and preserve all pre-existing initialization.
        self.key_quality_projection = nn.Parameter(torch.zeros(heads, 2))

    def forward(
        self,
        tokens: torch.Tensor,
        key_log_support: torch.Tensor,
        key_quality_features: torch.Tensor,
        value_strength: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, token_count, dim = tokens.shape
        if token_valid_mask.dtype != torch.bool or token_valid_mask.shape != (batch, token_count):
            raise ValueError("token_valid_mask must be bool with shape (B, N)")
        if token_valid_mask.device != tokens.device or not bool(token_valid_mask[:, 0].all()):
            raise ValueError("token_valid_mask must share the device and retain CLS")
        valid = token_valid_mask.unsqueeze(-1)
        tokens = tokens.masked_fill(~valid, 0.0)
        expected_strength_shape = (batch, token_count, 1)
        if key_log_support.shape != expected_strength_shape:
            raise ValueError(
                "key_log_support must have shape "
                f"{expected_strength_shape}, got {tuple(key_log_support.shape)}"
            )
        if key_quality_features.shape != (batch, token_count, 2):
            raise ValueError(
                "key_quality_features must have shape "
                f"{(batch, token_count, 2)}, got {tuple(key_quality_features.shape)}"
            )
        if value_strength.shape != expected_strength_shape:
            raise ValueError(
                f"value_strength must have shape {expected_strength_shape}, "
                f"got {tuple(value_strength.shape)}"
            )

        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).reshape(
            batch, token_count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        value_scalar = value_strength.squeeze(-1).to(dtype=tokens.dtype)
        gate = self.value_gate_floor + (1.0 - self.value_gate_floor) * value_scalar
        value = value * gate[:, None, :, None]
        value = value.masked_fill(~token_valid_mask[:, None, :, None], 0.0)

        key_log_prior = key_log_support.squeeze(-1).to(dtype=tokens.dtype)
        if not bool(torch.isfinite(key_log_prior).all()):
            raise ValueError("key log-support must be finite")
        if self.attention_alpha == 0.0:
            attention_bias = (
                self.attention_alpha * key_log_prior
            )[:, None, None, :]
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_bias.masked_fill(
                    ~token_valid_mask[:, None, None, :], -torch.inf
                ),
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            quality_key_logit = F.linear(
                key_quality_features.float(), self.key_quality_projection.float()
            ).to(dtype=tokens.dtype).transpose(1, 2)
            # A per-head [B,H,1,N] float mask makes CUDA SDPA materialize the
            # broadcast [B,H,N,N] bias at the production token count. Encode
            # the same additive Key logit in eight aligned Q/K channels so the
            # only explicit mask remains the compact shared padding mask.
            # Store fixed and learned biases separately to avoid losing the
            # small learned term when their sum is cast to BF16. The existing
            # scale makes each extra (1, bias*sqrt(head_dim)) pair add bias.
            padding_width = 8
            query_padding = torch.zeros(
                (*query.shape[:-1], padding_width),
                dtype=query.dtype,
                device=query.device,
            )
            query_padding[..., :2] = 1.0
            key_padding = torch.zeros_like(query_padding)
            key_padding[..., 0] = (
                self.attention_alpha * key_log_prior[:, None, :]
            ) * math.sqrt(self.head_dim)
            key_padding[..., 1] = (
                self.attention_alpha * quality_key_logit
            ) * math.sqrt(self.head_dim)
            value_padding = torch.zeros_like(query_padding)
            query_augmented = torch.cat((query, query_padding), dim=-1)
            key_augmented = torch.cat((key, key_padding), dim=-1)
            value_augmented = torch.cat((value, value_padding), dim=-1)
            padding_bias = torch.zeros_like(
                key_log_prior[:, None, None, :]
            ).masked_fill(~token_valid_mask[:, None, None, :], -torch.inf)
            attended = F.scaled_dot_product_attention(
                query_augmented,
                key_augmented,
                value_augmented,
                attn_mask=padding_bias,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(self.head_dim),
            )[..., : self.head_dim]
        attended = attended.transpose(1, 2).reshape(batch, token_count, dim)
        tokens = tokens + self.projection(attended)
        return (tokens + self.mlp(self.norm2(tokens))).masked_fill(~valid, 0.0)
