"""Support-dependent scaling of optical tokens before the Transformer."""

import torch
import torch.nn as nn


class ReliabilityTokenGate(nn.Module):
    """Scale each token by gate_min + (1 - gate_min) * support."""

    def __init__(self, gate_min: float = 0.1):
        super().__init__()
        self.gate_min = gate_min

    def forward(self, tokens: torch.Tensor, r_patch: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must be (B,T,L,D)")
        if r_patch.ndim != 4 or r_patch.shape[:3] != tokens.shape[:3] or r_patch.shape[-1] != 1:
            raise ValueError("r_patch must be (B,T,L,1) and align with tokens")
        gate = self.gate_min + (1.0 - self.gate_min) * r_patch
        return tokens * gate
