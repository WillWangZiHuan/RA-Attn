# RA-Attn

Selected method source for **Reliability-Aware Attention** in grassland NDVI forecasting.

> **Partial source release for reading. Training and inference workflows have not been released.**
>
> **Full code and expanded global grassland dataset — coming soon.**

## Current release: v35 development snapshot

The current source excerpts come from the verified v35 implementation used by the 2026-09-13 experiment package. They are selected parts of a development implementation; they do not constitute a complete model release or identify a final selected experiment configuration.

| File | Included component |
| --- | --- |
| [`source/attention.py`](source/attention.py) | `ReliabilityAttentionBlock`: separate key-support, learned per-head key features, value gating, padding masks, and the augmented Q/K computation used to avoid an explicit per-head attention-bias matrix. |
| [`source/history_anomaly.py`](source/history_anomaly.py) | `observed_ndvi_history_anomaly`: deviations from the observed historical NDVI mean, excluding missing observations and padded years. |

The selected class and function are copied verbatim from the verified source file. Only module headers and the imports needed by these excerpts have been added. The full source file has SHA-256:

`6ed0d446328b42495b4936be361ae07e22c91699d3b4699358cc2eeefef7862a`

The complete v35 implementation uses 4×4 spatial tokens, distinct observation-count and QA-retention inputs, a history-NDVI anomaly embedding, and residual prediction added to same-pixel historical climatology. The `old` and `split_kv` variants share this implementation but differ in how support determines value strength. The full tokenization, support routing, anomaly projection, climatology construction, readout, and model assembly are **not included** in these excerpts.

Training scripts, inference scripts, preprocessing, experiment configurations, model checkpoints, and datasets are also **not included in this release**. This repository provides no end-to-end runnable example and is not sufficient to reproduce the reported experiments.

## Reading the excerpts

The attention block receives tokens `(B, N, D)`, key log-support `(B, N, 1)`, two key-quality features `(B, N, 2)`, value strength `(B, N, 1)`, and a Boolean valid-token mask `(B, N)`. Here, `B` is batch size, `N` is sequence length, and `D` is embedding dimension. The caller constructs these inputs and retains the class token in the valid-token mask.

The history-anomaly function receives optical features `(B, T, 8, H, W)`, support `(B, T, 3, H, W)`, and a Boolean year mask `(B, T)`. It reads the NDVI feature at optical channel 7 and the observed-history indicator at support channel 2. Its output is the observed deviation in the input NDVI normalization units. The caller validates and constructs the inputs; this function does not load data or produce a forecast.

Observation support and QA retention describe the inputs. They are not calibrated prediction uncertainty. The learned key-feature projection in this v35 excerpt introduces trainable parameters; this development snapshot should not be described as the earlier parameter-free conditioning formulation.

## Release plan — coming soon

**Remaining code.** We are organizing and documenting the rest of the project. We will upload the full project code, including the training and inference workflows, once this work is complete.

**Dataset.** We are expanding the dataset toward coverage of grasslands worldwide. Once the expansion to global grassland coverage and data organization are complete, we will also release the expanded dataset through this repository alongside the full code. The dataset is not currently available for download here.

Both releases are in preparation. A release date has not yet been finalized; updates will be posted in this repository.

## 中文说明

**本仓库目前仅公开选定的部分方法源码，供阅读和理解。训练和推理流程尚未发布，暂不提供可直接运行的完整实例。**

当前公开内容仅包括今天运行包所使用的 v35 注意力模块和历史 NDVI 异常计算片段，不包含完整模型、训练及推理脚本、数据处理流程、实验配置、模型权重或数据集。当前仓库不足以完整复现论文实验，也不表示最终模型配置已经选定。

其余代码仍在整理和补充说明，整理完成后将全部上传，包括完整的训练与推理流程。

我们的数据集也在持续扩充，目标是覆盖全球草原。待全球草原范围的扩充及数据整理完成后，扩充后的数据集也将与完整代码一并通过本仓库发布。

**完整代码与全球草原数据集：Coming soon。具体发布时间尚未确定，请关注本仓库更新。**
