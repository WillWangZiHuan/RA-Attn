# RA-Attn

Reliability-Aware Attention for grassland NDVI forecasting.

> **Core attention module available now. Full code and expanded global grassland dataset — coming soon.**

## Included core module

[`source/attention.py`](source/attention.py) contains `TransformerBlock`, the reliability-aware attention block used in the current Landsat-based working revision. The block is retained from the first-revision source family and has no precipitation input.

The module includes the complete block implementation: query/key/value projections, a reliability-dependent bias on attention keys, reliability-dependent value gating, and the residual and feed-forward operations. Its computational logic and constructor defaults match the corresponding local source; only the module header and comments were edited for reading.

## Interface

The block receives token embeddings `x` with shape `(B, N, D)` and aligned reliability values `r_all` with shape `(B, N, 1)`, and returns updated embeddings with shape `(B, N, D)`. Reliability must be finite and lie in `[0, 1]`. Here, `B` is batch size, `N` is token count, and `D` is embedding width. The caller constructs and validates these inputs.

The constructor exposes `alpha` for the key bias and `gate_min` for the value gate. The caller specifies the parameters for each experiment. The optional `collect_influence` flag records attention averaged over heads and queries.

## Code and dataset release plan

Thank you for your interest in RA-Attn. This repository currently shares the core reliability-aware attention module for research and reference. We are organizing the remaining project code and accompanying documentation, including the complete model and the training and inference workflows.

Alongside this work, we are expanding the dataset toward coverage of grasslands worldwide. Once the code organization and dataset expansion are complete, we plan to release the complete codebase together with the expanded dataset through this repository.

Release updates will be announced here as the work progresses. Thank you for your patience and support.

**Full code and expanded global grassland dataset — coming soon.**
