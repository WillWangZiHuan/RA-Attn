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

## 中文说明

感谢您对 RA-Attn 的关注！本仓库目前分享核心可靠性感知注意力模块，供研究交流与参考。

`source/attention.py` 包含可靠性感知注意力块的完整实现，涵盖注意力 Key 偏置和 Value 门控。该模块来自一修时期源码，并沿用于当前移除降水分支的 Landsat 修改版；公开模块本身不含降水输入。

我们正在进一步整理其余代码及配套文档，包括完整模型和训练、推理流程。同时，数据集也在持续扩充，目标是覆盖全球草原。待相关整理与扩充工作完成后，我们计划将完整代码与扩充后的数据集一并公开发布。

后续进展与发布信息将在本仓库持续更新。感谢您的耐心等待与支持！

**完整代码与扩充后的全球草原数据集：Coming soon。**
