# Image Recognition Methods

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Laboratory implementations and experiments covering foundational and modern computer vision methods: classical feature extraction with edge filters, residual convolutional networks with hyperparameter optimization, deep generative modeling (DCGAN and DDPM), real-time object detection with YOLOv8, and Vision Transformers with attention rollout.

<p align="center">
  <img src="./lab5/extra/bike.gif" alt="DDPM Reverse Diffusion on Bike Distribution" width="480">
</p>

## Overview

<details open>
<summary><b>Lab 1: Canny Edge Detector in tinygrad</b></summary>

- **Focus**: Classical edge detection, custom convolution kernels, and tensor execution in tinygrad.
- **Key Concepts**:
  - RGB-to-grayscale transformation and normalized tensor representation on GPU.
  - 2D Gaussian smoothing convolution filter.
  - Sobel spatial gradient operators for magnitude estimation and gradient angle discretization.
  - Non-maximum suppression (NMS) along estimated edge normals.
  - Double-threshold filtering and hysteresis edge tracking.
- **Directory**: [`lab1/`](./lab1/)
- **Script**: [`lab1/lab1.py`](./lab1/lab1.py)
</details>

<details>
<summary><b>Lab 2: ResNet-9 Optimization on CIFAR-10</b></summary>

- **Focus**: Deep residual convolutional architectures, regularization, and automated hyperparameter optimization.
- **Key Concepts**:
  - Custom ResNet-9 modular convolutional architecture with residual skip connections (`x + res(x)`).
  - Batch normalization, spatial max pooling, and dropout regularization.
  - Systematic Bayesian hyperparameter search with **Optuna** over learning rates, weight decay, and dropout rates.
  - PyTorch training and validation loops with model checkpointing.
- **Directory**: [`lab2/`](./lab2/)
- **Script & Notebook**: [`lab2/lab2.py`](./lab2/lab2.py) & [`lab2/lab2.ipynb`](./lab2/lab2.ipynb)
</details>

<details>
<summary><b>Lab 3: DCGAN on Custom Image Collections</b></summary>

- **Focus**: Generative adversarial networks, min-max training dynamics, and latent space sampling.
- **Key Concepts**:
  - Deep Convolutional GAN (DCGAN) generator with transposed convolutions and batch normalization.
  - Strided convolutional discriminator with binary cross-entropy with logits.
  - Adversarial training stability using AdamW with modified momentum ($\beta_1 = 0.5$).
  - Qualitative evaluation and loss curve tracking on a custom crawled carrot cake dataset (`crawled_cakes.tar.gz`).
- **Directory**: [`lab3/`](./lab3/)
- **Notebook**: [`lab3/lab3.ipynb`](./lab3/lab3.ipynb)
</details>

<details>
<summary><b>Lab 4: Object Detection with YOLOv8 & Open Images</b></summary>

- **Focus**: Real-time object detection, dataset curation, and transfer learning.
- **Key Concepts**:
  - Zero-shot inference and validation using pretrained YOLOv8 (`ultralytics`).
  - Automated subset filtering from Open Images V7 annotations (Food category) and conversion to YOLO bounding box format (`[class, xc, yc, w, h]`).
  - Transfer learning and fine-tuning YOLOv8m on target categories.
  - Downstream postprocessing: bounding box visualization and selective background blurring.
- **Directory**: [`lab4/`](./lab4/)
- **Script**: [`lab4/lab4.py`](./lab4/lab4.py)
</details>

<details>
<summary><b>Lab 5: Denoising Diffusion Probabilistic Model (DDPM)</b></summary>

- **Focus**: Score-based generative models, diffusion variance scheduling, and reverse trajectory sampling.
- **Key Concepts**:
  - DDPM implemented from scratch in PyTorch on a 2D point cloud distribution (`bicycle.txt`).
  - Analytical and numerical evaluation of linear vs. sigmoidal variance schedules ($\beta_t$, $\bar{\alpha}_t$).
  - Learnable sinusoidal time step embeddings for conditioning denoising networks.
  - Forward diffusion with multivariate normal noise injection and iterative reverse denoising trajectory generation (`bike.gif`).
- **Directory**: [`lab5/`](./lab5/)
- **Script**: [`lab5/lab5.py`](./lab5/lab5.py)
</details>

<details>
<summary><b>Lab 6: Vision Transformer (ViT) & Attention Rollout</b></summary>

- **Focus**: Self-attention in vision, patch embeddings, and model interpretability.
- **Key Concepts**:
  - Vision Transformer (ViT) implemented from scratch in PyTorch for CIFAR-10 classification.
  - 2D image patch partitioning (`Patchify`) with learnable linear patch projections.
  - Learnable 1D position embeddings and `[CLS]` token concatenation.
  - Multi-Head Self-Attention (MHSA) encoder blocks with Pre-LN LayerNorm and residual connections.
  - Attention rollout technique to project self-attention matrices back to input pixel space for visual interpretability.
- **Directory**: [`lab6/`](./lab6/)
- **Script**: [`lab6/lab6.py`](./lab6/lab6.py)
</details>

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/barhanc/image-recognition-methods.git
cd image-recognition-methods
```

### 2. Environment Setup

Manage dependencies and runtime environments using [uv](https://docs.astral.sh/uv/):

```bash
# Synchronize environment and install dependencies from lockfile
uv sync
```

### 3. Running Labs

Execute standalone scripts or launch a Jupyter environment:

```bash
# Run a specific lab script
uv run python lab1/lab1.py
uv run python lab5/lab5.py

# Launch Jupyter Lab for interactive notebooks
uv run jupyter lab
```
