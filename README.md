# RA-Attn

Selected method source for **Reliability-Aware Attention** in grassland NDVI forecasting.

> **Partial source release for reading. Training and inference workflows have not been released.**
>
> **Full code and expanded global grassland dataset — coming soon.**

## Current release

This public repository currently contains only selected implementation excerpts for understanding the attention and fusion operations. It is an incomplete code release and does not provide an end-to-end runnable example or the materials needed to reproduce the reported experiments.

| File | Included component |
| --- | --- |
| [`source/attention.py`](source/attention.py) | A Transformer block with a centered log-support bias on the key axis and support-dependent value gating. |
| [`source/fusion.py`](source/fusion.py) | Feature concatenation, linear projection, and support gating before the Transformer. |

The full forecasting model, encoders, prediction head, support-construction and data-processing pipeline, training scripts, inference scripts, experiment configurations, model checkpoints, and datasets are **not included in this release**.

## Reading the excerpts

The attention block receives tokens of shape `(B, N, D)` and aligned support values of shape `(B, N, 1)`. The fusion component receives two token tensors of shape `(B, T, L, D)` and support of shape `(B, T, L, 1)`. Here, `B` is batch size, `N` is sequence length, `D` is embedding dimension, `T` is the number of input time steps, and `L` is the number of spatial tokens per time step.

Support values are supplied by the caller in `[0, 1]`; the support-construction pipeline is not published here. When a class token is present, its aligned support is 1. The support signal is an observation-count proxy, not calibrated observation quality or predictive uncertainty.

These excerpts retain the computations and generic constructor defaults of the selected source components; comments and docstrings have been edited for readability. The manuscript configuration sets the key-bias coefficient to `alpha=0.5`, the fusion-gate floor to `gate_min=0.1`, and the attention value-gate floor to `gate_min=0.3`. The generic defaults in the two classes are not a complete experiment configuration.

## Release plan — coming soon

**Remaining code.** We are organizing and documenting the rest of the project. We will upload the full project code, including the training and inference workflows, once this work is complete.

**Dataset.** We are expanding the dataset toward coverage of grasslands worldwide. Once the expansion to global grassland coverage and data organization are complete, we will also release the expanded dataset through this repository alongside the full code. The dataset is not currently available for download here.

Both releases are in preparation. A release date has not yet been finalized; updates will be posted in this repository.

## 中文说明

**本仓库目前仅公开选定的部分方法源码，供阅读和理解。训练和推理流程尚未发布，暂不提供可直接运行的完整实例。**

当前公开内容仅包括注意力模块和融合模块的局部实现，不包含完整模型、训练及推理脚本、数据处理流程、实验配置、模型权重或数据集。当前仓库不足以完整复现论文实验。

其余代码仍在整理和补充说明，整理完成后将全部上传，包括完整的训练与推理流程。

我们的数据集也在持续扩充，目标是覆盖全球草原。待全球草原范围的扩充及数据整理完成后，扩充后的数据集也将与完整代码一并通过本仓库发布。

**完整代码与全球草原数据集：Coming soon。具体发布时间尚未确定，请关注本仓库更新。**
