# RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models

[![ArXiv](https://img.shields.io/badge/ArXiv-2511.19704-b31b1b.svg)](https://arxiv.org/abs/2511.19704)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **RADSeg**, a framework leveraging the agglomerative vision foundation model **RADIO** to improve zero-shot Open-Vocabulary Semantic Segmentation (OVSS). RADSeg enhances performance through self-correlating recursive attention, global aggregation, and efficient mask refinement, achieving state-of-the-art results with significantly lower computational and memory costs compared to existing methods.

## Introduction

![RADSeg Architecture](assets/RADSeg-Diagram.png)
*Figure 1: Overview of the RADSeg architecture.*

Existing SOTA OVSS approaches often rely on heavy combinations of multiple models (e.g., CLIP + DINO + SAM). **RADSeg** introduces a unified, parameter and compute efficient approach by adapting **RADIO** for zero-shot open vocabulary segmentation.

**Key Features:**
- **Parameter Efficiency**: 2.5x fewer parameters than comparable state-of-the-art methods.
- **Speed**: ~4x faster inference.
- **Performance**: Significant mIoU improvements (6-30% on base ViT class) across benchmarks.
- **Unified Backbone**: Uses RADIO as a single powerful backbone for both semantic and spatial understanding.



## Usage

### Evaluation

To evaluate RADSeg on a specific dataset:

```bash
python eval.py \
  --config configs_mmseg/YOUR_CONFIG.py \
  --model_version radio_v3-b \
  --lang_model siglip2 \
  --scra_scaling 10 \
  --scga_scaling 10.0 \
  --work-dir ./work_logs/ \
  --sam_refine True
```

Arguments:
- `--config`: Path to the mmseg config file.
- `--model_version`: RADIO model version (e.g., `radio_v3-b`).
- `--lang_model`: Language model to use (e.g., `siglip2`).
- `--scra_scaling`: Scaling factor for Self-Correlating Recursive Attention (SCRA).
- `--scga_scaling`: Scaling factor for Semantic Consensus Global Aggregation (SCGA).
- `--sam_refine`: Enable SAM-based mask refinement (optional, set to True).


### Evaluation across Different Resolutions and Datasets

To run evaluation across multiple resolutions and configs as defined in `eval_all_multi_res.py`:

```bash
python eval_all_multi_res.py
```
This script iterates over defined configurations (Low Resolution, Mid Resolution and High Resolution) and runs the evaluation automatically.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{alama2025radseg,
  title={RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models},
  author={Alama, Omar and Jariwala, Darshil and Bhattacharya, Avigyan and Kim, Seungchan and Wang, Wenshan and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2511.19704},
  year={2025}
}
```

## Acknowledgements

This codebase is built upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [Trident](https://github.com/YuHengsss/Trident). We thank the authors for their open-source contributions.
