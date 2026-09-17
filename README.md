# RA-Attn

Reliability-Aware Attention for grassland NDVI-change forecasting.

> **Complete model architecture available now. The remaining project code and expanded global grassland dataset are coming soon.**

## Model architecture

This repository provides the complete optical-only RA-Attn model, from prepared Landsat inputs and QA-retained observation counts to a predicted delta-NDVI map. The implementation includes the optical encoder, support construction, token gate, spatiotemporal Transformer, reliability-aware attention, and spatial prediction head.

The model uses non-overlapping 8 x 8 pixel patches, an embedding width of 256, six Transformer blocks, and eight attention heads. Each 128 x 128 image produces 256 spatial tokens. Twelve annual inputs and one class token form a sequence of 3,073 tokens. The model has **4,940,352 trainable parameters**.

The main configuration uses a key-bias coefficient of 0.5, a token-gate floor of 0.1, and a value-gate floor of 0.3.

## Source files

| File | Component |
| --- | --- |
| [source/model.py](source/model.py) | `RAAttn`: complete model assembly and `NDVIMapHead`. |
| [source/encoders.py](source/encoders.py) | Convolutional Landsat patch embedding. |
| [source/reliability.py](source/reliability.py) | Per-sample, per-year retained-count normalization and token pooling. |
| [source/token_gate.py](source/token_gate.py) | Support-dependent scaling before the Transformer. |
| [source/backbone.py](source/backbone.py) | Spatial and temporal embeddings, class token, and six-block backbone. |
| [source/attention.py](source/attention.py) | Reliability-dependent key bias and value gating. |
| [source/__init__.py](source/__init__.py) | Public `RAAttn` import. |

## Model interface

Import the model with `from source import RAAttn`. Construct it with `RAAttn(alpha=0.5, tau_f=0.1, tau_v=0.3)` and call `model(optical, retained)`.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `optical` | `(B, 12, 7, 128, 128)` | Prepared, scaled Landsat surface reflectance in chronological order. |
| `retained` | `(B, 12, 1, 128, 128)` | Aligned QA-retained observation counts. |
| Output | `(B, 1, 128, 128)` | Predicted next-year NDVI minus the last input year's NDVI. |

Inputs must be finite floating-point tensors on the model's device, with nonnegative retained counts. For the float32 model, use float32 inputs. Historical sequences shorter than 12 years are right-aligned; unused earlier slots have zero optical values and zero counts. Input preparation supplies numerical zeros for invalid optical values.

For each sample and year, pixel support is `retained / max(1, spatial_max(retained))`. Non-overlapping 8 x 8 average pooling gives token support. An all-zero count map produces zero support. This signal describes relative retained-count support within a patch and year.

The same token support controls the token gate, key bias, and value gate. Spatial and temporal positional embeddings are added after the token gate. The class token has support 1. After the Transformer, the class token is removed and the final-year spatial tokens are decoded into the output map. The temporal embedding table retains its experimental capacity of 16 slots; the public forecasting interface uses 12 input slots.

The output is an NDVI **change**. Adding the observed NDVI of the last input year yields the corresponding next-year NDVI prediction.

## Dependencies and verification

The model depends on PyTorch. [requirements.txt](requirements.txt) records the version used for release verification.

This release is extracted and reorganized from the optical-only model used in the current experiments. CPU checks with PyTorch 2.10.0 confirmed identical parameter names, initialization, forward outputs, and gradients against that implementation using the same inputs and weights. Synthetic checks covered mixed, all-zero, and uniform retained counts, padded historical slots, and an alternative set of gate coefficients. The source package contains the complete model computation and uses standard PyTorch modules.

## Code and dataset release plan

Thank you for your interest in RA-Attn. We are organizing the remaining project code and accompanying documentation, including data preparation, training, evaluation, and inference workflows, together with trained model weights.

Alongside this work, we are expanding the dataset toward coverage of grasslands worldwide. Once the code organization and dataset expansion are complete, we plan to release the remaining implementation and model resources together with the expanded dataset through this repository.

Release updates will be announced here as the work progresses. Thank you for your patience and support.

**Remaining project code and expanded global grassland dataset — coming soon.**
