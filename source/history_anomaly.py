"""Selected v35 history-anomaly operation for source reading.

This is the input-side NDVI anomaly operation only. The projection layer,
complete forecasting model, data loader and training/inference workflows
are not included in this partial release.
"""

from __future__ import annotations

import torch


def observed_ndvi_history_anomaly(
    optical: torch.Tensor, support: torch.Tensor, year_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return observed NDVI deviations; retain raw optical in its original path.

    Called after the native input validation. All means use only the supplied
    pre-target history. Missing observations and padded years contribute zero.
    The output is in the existing frozen normalized NDVI units. It is not an
    uncertainty estimate and introduces no new QA weighting or target input.
    """
    weights = support[:, :, 2:3].masked_fill(
        ~year_valid_mask[:, :, None, None, None], 0.0
    ).to(dtype=optical.dtype)
    present = weights > 0
    ndvi = optical[:, :, 7:8].masked_fill(~present, 0.0)
    count = weights.sum(dim=1, keepdim=True)
    denominator = torch.where(count > 0, count, torch.ones_like(count))
    mean = (ndvi * weights).sum(dim=1, keepdim=True) / denominator
    return (ndvi - mean).masked_fill(~present, 0.0)
