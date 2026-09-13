# RA-Attn

Reliability-Aware Attention for grassland NDVI forecasting.

> **Partial supporting source for reading. The complete method, training and inference workflows have not been released. The current release is insufficient to reproduce the reported experiments.**
>
> **Full code and expanded global grassland dataset — coming soon.**

## Current public content

This release contains two supporting components from the project's Landsat-based working code:

| File | Included component |
| --- | --- |
| [`source/patch_embedding.py`](source/patch_embedding.py) | `LandsatPatchEncoder`: convolutional patch embedding of a Landsat sequence. |
| [`source/reliability_pooling.py`](source/reliability_pooling.py) | `ReliabilityBuilder`: conversion of an already normalized quality-deficit tensor to pixel support and average-pooled patch support. |

The included computational logic matches the corresponding working-source components. Comments and documentation have been edited for reading. These utilities alone do not implement RA-Attn or produce NDVI forecasts. This snapshot does not establish an exact source version for any reported experiment.

The complete attention mechanism and model assembly, data preparation, training and inference scripts, experiment configurations, model weights, and datasets are **not included in this release**. No end-to-end runnable example or experiment reproduction package is currently provided.

## Component interfaces

`LandsatPatchEncoder` maps `(B, T, C, H, W)` to `(B, T, L, D)` with a convolution whose kernel and stride equal the supplied patch size. `ReliabilityBuilder` takes a finite quality-deficit tensor named `cloud`, shaped `(B, T, 1, H, W)`, with values in `[0, 1]`; larger values indicate lower support. It returns pixel support and average-pooled patch support. This input is already normalized by the caller; it is not a raw observation-count raster or a raw QA bitfield.

Here, `B` is batch size, `T` is the number of time steps, `C` is the number of input channels, `D` is embedding width, and `L` is the number of spatial patches. Spatial dimensions are assumed to be divisible by the supplied patch size. The supporting utilities do not specify the complete model or experiment settings.

## Remaining code — coming soon

The remaining project code is being revised and organized. We plan to upload the full code, including training and inference workflows, once the code and accompanying documentation are ready. A release date has not yet been finalized.

## Dataset availability

We are expanding the dataset toward coverage of grasslands worldwide. Once the expansion to global grassland coverage and data organization are complete, the expanded dataset will also be released through this repository alongside the full code.

The dataset is not currently available for download here. A release date has not yet been finalized. Updates will be posted in this repository.

## 中文说明

**本仓库仅提供部分辅助模块源码，供阅读。完整方法及训练和推理流程尚未发布，当前内容不足以复现论文实验。**

当前公开内容包括 Landsat patch 编码和支持度池化两个辅助模块。完整注意力机制、模型组装、数据处理、训练与推理脚本、实验配置、模型权重和数据集均不在本次发布范围内；暂不提供端到端可运行实例。该快照不表示已公开与论文某项实验精确对应的完整源码版本。

其余代码正在修改和整理，计划在代码及说明整理完成后全部上传，包括完整的训练与推理流程。

我们的数据集也在持续扩充，目标是覆盖全球草原。待全球草原范围的扩充及数据整理完成后，扩充后的数据集也将与完整代码一并通过本仓库发布。

**完整代码与全球草原数据集：Coming soon。具体发布时间尚未确定，请关注本仓库更新。**
