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

### [Lab 01: Canny Edge Detector in tinygrad](./lab1/)
- **Core Topics**: Classical edge detection, custom convolution kernels, and tensor execution in tinygrad.
- **Implementation**: Grayscale transformation, 2D Gaussian smoothing filter, Sobel spatial gradient operators for magnitude and gradient angle estimation, non-maximum suppression (NMS), and double-threshold hysteresis edge tracking.

### [Lab 02: ResNet-9 Optimization on CIFAR-10](./lab2/)
- **Core Topics**: Deep residual networks, learning rate scheduling, and automated hyperparameter tuning.
- **Implementation**: Custom ResNet-9 convolutional architecture with residual skip connections, batch normalization, max pooling, and dropout. Hyperparameter search executed with Optuna across learning rates, weight decay, and dropout rates.

### [Lab 03: DCGAN on Custom Image Collections](./lab3/)
- **Core Topics**: Generative adversarial networks, adversarial training dynamics, and latent space sampling.
- **Implementation**: Deep Convolutional GAN (DCGAN) with a transposed convolutional Generator and strided convolutional Discriminator. Trained with AdamW and binary cross-entropy on a curated dataset of carrot cakes (`crawled_cakes.tar.gz`).

### [Lab 04: Object Detection with YOLOv8 & Open Images](./lab4/)
- **Core Topics**: Real-time object detection, transfer learning, and dataset curation.
- **Implementation**: Zero-shot inference and fine-tuning with YOLOv8 (`ultralytics`). Automated subset extraction from Open Images V7 annotations (Food class), conversion to YOLO bounding box format, model fine-tuning, and inference with bounding box rendering and background blurring.

### [Lab 05: Denoising Diffusion Probabilistic Model (DDPM)](./lab5/)
- **Core Topics**: Score-based generative models, diffusion variance scheduling, and reverse trajectory sampling.
- **Implementation**: DDPM implemented from scratch in PyTorch on a 2D point cloud distribution (`bicycle.txt`). Compares linear and sigmoidal variance schedules ($\beta_t$, $\bar{\alpha}_t$), learnable sinusoidal time step embeddings, forward multivariate normal diffusion, and iterative reverse denoising synthesis.

### [Lab 06: Vision Transformer (ViT) & Attention Rollout](./lab6/)
- **Core Topics**: Self-attention in vision, patch embeddings, and interpretability.
- **Implementation**: Vision Transformer built from scratch in PyTorch for CIFAR-10 classification. Implements 2D image patch partitioning (`Patchify`), learnable linear projections, `[CLS]` token and positional embeddings, stacked Multi-Head Self-Attention (MHSA) encoder blocks with LayerNorm, and attention rollout to visualize receptive fields in input pixel space.

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
