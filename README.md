<h1 align="center">RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models</h1>

<p align="center">
  <a href="https://oasisartisan.github.io/"><strong>Omar Alama</strong></a>*
  ·
  <a href="https://www.linkedin.com/in/darshil-jariwala"><strong>Darshil Jariwala</strong></a>*
  ·
  <a href="https://avigyanbh.github.io/"><strong>Avigyan Bhattacharya</strong></a>*
  <br>
  <a href="https://seungchan-kim.github.io/"><strong>Seungchan Kim</strong></a>
  ·
  <a href="https://theairlab.org/team/wenshan/"><strong>Wenshan Wang</strong></a>
  ·
  <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
</p>

  <h3 align="center"><a href="https://arxiv.org/abs/2511.19704">Paper</a> | <a href="https://radseg-ovss.github.io/">Project Page</a> | <a href="#">Demo</a></h3>
  <div align="center"></div>


This repository contains the official implementation of **RADSeg**, a framework leveraging the agglomerative vision foundation model **RADIO** to improve zero-shot Open-Vocabulary Semantic Segmentation (OVSS). RADSeg enhances performance through self-correlating recursive attention, global aggregation, and efficient mask refinement, achieving state-of-the-art results with significantly lower computational and memory costs compared to existing methods.

## Introduction

![RADSeg Architecture](assets/abstract_figure.svg)

Existing SOTA OVSS approaches often rely on heavy combinations of multiple models (e.g., CLIP + DINO + SAM). **RADSeg** introduces a unified, parameter and compute efficient approach by adapting **RADIO** for zero-shot open vocabulary segmentation.

**Key Features:**
- **Unified Backbone**: Uses RADIO as a single powerful vision backbone for zero-shot open vocabulary semantic segmentation.
- **Efficiency**: 3.9x faster inference and 2.5x fewer parameters than comparable state-of-the-art methods.
- **Performance**: Significant mIoU improvements (6-30% on base ViT class) across benchmarks.

## Environment Setup

Create a conda environment and install base dependencies:
```bash
conda env create -f environment.yml
conda activate radseg
```

Additional dependencies for 2D evaluation:
1. Install OpenMMLab dependencies:
   ```bash
   pip install mmengine==0.10.1
   pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
   pip install mmsegmentation==1.2.2
   ```

2. **MMSegmentation Compatibility:** 
   In `{site-packages-path}/mmseg/__init__.py`, you may need to update the `mmcv` version check (Approved by original mmcv author). Change:
   ```python
   assert (mmcv_min_version <= mmcv_version < mmcv_max_version)
   ```
   to:
   ```python
   assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)
   ```

Additional dependencies for 3D evaluation:
Please follow the minimal setup instructions of [RayFronts Environment Setup](https://github.com/RayFronts/RayFronts?tab=readme-ov-file#environment-setup) to set up the conda/mamba environment for 3D evaluations. 


## Quickstart

### Torch Hub
You can easily load RADSeg using Torch Hub for integration into your own projects:

```python
import torch
from PIL import Image
import torchvision.transforms as T

# Load RADSeg model
model = torch.hub.load('RADSeg-OVSS/RADSeg', 'radseg_encoder', 
                       model_version="c-radio_v3-b", 
                       lang_model="siglip2")
model.to('cuda').eval()

# Prepare image
img = Image.open('your_image.jpg').convert('RGB')
img_tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')

# Define labels for zero-shot segmentation
labels = ["sky", "grass", "cat", "tree"]

# High-level API for segmentation
model.predict = True
model.prompts = labels
model.text_embeds = model.encode_labels(labels)
with torch.no_grad():
    seg_probs = model.encode_image_to_feat_map(img_tensor) # [1, len(labels)+1, H, W]
```

### Gradio Demo
To test RADSeg on your own images using an interactive Gradio interface:

1. **Activate Environment**:
   ```bash
   conda activate radseg
   ```

2. **Run the App**:
   ```bash
   python radseg_demo.py
   ```
This will launch an interface where you can upload images, add custom text prompts, and adjust model parameters


## 2D Evaluation

### Dataset Preparation
Please follow the [MMSegmentation data preparation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and process the 5 2D datasets.

### Running Evaluation 
To evaluate RADSeg on a specific 2D dataset, switch to the evaluation/2d directory and run:

```bash
python eval.py \
  --config configs_mmseg/YOUR_CONFIG.py \
  --model_version c-radio_v3-b \
  --lang_model siglip2 \
  --scra_scaling 10.0 \
  --scga_scaling 10.0 \
  --work-dir ./work_logs/ \
  --sam_refine
```

Arguments:
- `--config`: Path to the mmseg config file.
- `--model_version`: RADIO model version (e.g., `c-radio_v3-b`).
- `--lang_model`: Language model to use (e.g., `siglip2`).
- `--scra_scaling`: Scaling factor for Self-Correlating Recursive Attention (SCRA).
- `--scga_scaling`: Scaling factor for Self-Correlating Global Aggregation (SCGA).
- `--sam_refine`: Enable RADIO-SAM mask refinement for RADSeg+ performance (include flag to enable).


To run evaluation across multiple resolutions and configs as defined in `eval_all.py`:

```bash
python eval_all.py
```
This script iterates over defined configurations (Low Resolution, Mid Resolution and High Resolution) and runs the evaluation automatically.

## 3D Evaluation

### Dataset Preparation
Please follow the guidelines and dataset download links provided by [RayFronts Datasets](https://github.com/RayFronts/RayFronts/tree/main/rayfronts/datasets#datasets--data-sourcesstreams) to process and prepare the 3 datasets (Replica - NiceReplica version, ScanNet, ScanNet++) used for 3D evaluation.

### Running Evaluation 
TODO

Switch to evaluation/3d directory
```bash
PYTHONPATH="../../:$PYTHONPATH" python RayFronts/scripts/semseg_eval.py --config-dir ./configs/ --config-name replica_radseg.yaml dataset.path="XXX"
```

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

This codebase is built upon [AM-RADIO](https://github.com/NVlabs/RADIO), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Trident](https://github.com/YuHengsss/Trident), and [RayFronts](https://github.com/RayFronts/RayFronts). We thank the authors for their open-source contributions.
