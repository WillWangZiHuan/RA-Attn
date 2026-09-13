# RA-Attn

Reliability-Aware Attention for grassland NDVI forecasting.

> **Partial method source for reading: one core attention block. Training and inference workflows have not been released.**
>
> **Full code and expanded global grassland dataset — coming soon.**

## Included core module

[`source/attention.py`](source/attention.py) contains `TransformerBlock`, the reliability-aware attention block used in the current Landsat-based working revision. The block is retained from the first-revision source family and has no precipitation input.

The module includes the complete block implementation: query/key/value projections, a reliability-dependent bias on attention keys, reliability-dependent value gating, and the residual and feed-forward operations. Its computational logic and constructor defaults match the corresponding local source; only the module header and comments were edited for reading.

## Interface

The block receives token embeddings `x` with shape `(B, N, D)` and aligned reliability values `r_all` with shape `(B, N, 1)`, and returns updated embeddings with shape `(B, N, D)`. Reliability must be finite and lie in `[0, 1]`. Here, `B` is batch size, `N` is token count, and `D` is embedding width. The caller constructs and validates these inputs.

The constructor exposes `alpha` for the key bias and `gate_min` for the value gate. Its defaults describe the standalone block and are not a complete experiment configuration. The optional `collect_influence` flag records attention averaged over heads and queries.

## Release scope

Only this core block is included in the current source release. The complete forecasting model, token and reliability preparation, model assembly, prediction head, training and inference scripts, experiment configurations, model weights, and datasets have not been released here.

This repository currently provides no end-to-end runnable example and is insufficient to reproduce the reported experiments. This partial snapshot does not establish an exact source version for a particular paper submission or reported result.

## Remaining code — coming soon

The remaining code is being revised, organized, and documented. We plan to upload the full project code, including training and inference workflows, once this work is complete.

## Dataset availability

We are expanding the dataset toward coverage of grasslands worldwide. Once the expansion to global grassland coverage and data organization are complete, the expanded dataset will also be released through this repository alongside the full code.

The dataset is not currently available for download here. A release date for the full code and dataset has not yet been finalized. Updates will be posted in this repository.

## 中文说明

**本仓库当前仅公开一个核心注意力模块，供阅读和理解：部分源码，训练和推理流程尚未发布。**

`source/attention.py` 包含可靠性感知注意力块的完整实现，涵盖注意力 Key 偏置和 Value 门控。该模块来自一修时期源码，并沿用于当前移除降水分支的 Landsat 修改版；公开模块本身不含降水输入。

本次仅提供上述核心模块。完整预测模型、输入及可靠性构建、模型组装、预测头、训练与推理脚本、实验配置、模型权重和数据集暂未公开。当前内容不足以复现论文实验，也不表示已公开某次投稿或实验结果的完整源码快照。

其余代码仍在修改和整理，计划整理完成后全部上传，包括完整的训练与推理流程。我们的数据集也在持续扩充，目标是覆盖全球草原；待全球草原范围的扩充及数据整理完成后，扩充后的数据集将与完整代码一并上传。

**完整代码与全球草原数据集：Coming soon。具体发布时间尚未确定。**
