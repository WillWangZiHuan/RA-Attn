"""Selected pre-Transformer fusion component, released for source reading.

Encoders, support construction, full model assembly, and training/inference
workflows are not included in this partial release.
"""

import torch
import torch.nn as nn


class ReliabilityAwareFusion(nn.Module):
    """Project concatenated modality tokens and apply a linear support gate.

    The generic constructor default is gate_min=0.5. The manuscript
    configuration sets the fusion-gate floor to 0.1.
    """

    def __init__(self, embed_dim: int, gate_min: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_min = gate_min
        self.fusion_fc = nn.Linear(2 * embed_dim, embed_dim)

    def forward(
        self,
        ms_tokens: torch.Tensor,
        precip_tokens: torch.Tensor,
        r_patch: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse aligned modality tokens (B, T, L, D) using support (B, T, L, 1).

        The caller supplies finite, token-aligned support values in [0, 1].
        """
        if ms_tokens.shape != precip_tokens.shape:
            raise ValueError('ms_tokens 和 precip_tokens shape 必须一致（B,T,L,D）')
        if (
            r_patch.ndim != 4
            or r_patch.shape[:3] != ms_tokens.shape[:3]
            or r_patch.shape[-1] != 1
        ):
            raise ValueError('r_patch 需要是 (B,T,L,1)，并与 token 对齐')

        # Concatenate the two modalities, then project back to D channels.
        x = torch.cat([ms_tokens, precip_tokens], dim=-1)
        x = self.fusion_fc(x)

        # Apply the pre-Transformer support gate.
        gate = self.gate_min + (1.0 - self.gate_min) * r_patch
        x = x * gate
        return x
