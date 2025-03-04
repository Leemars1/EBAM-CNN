# EBAM-CNN: A Deep Learning Model for Thermocline Reconstruction

## Overview

EBAM-CNN is a deep learning model designed for oceanographic research, specifically aimed at downscaling thermocline depth. The model incorporates attention mechanisms, including **Channel Attention** and **Spatial Attention**, to enhance performance in predicting spatial and temporal variations of oceanic features.

## Features

- **EBAM (Enhanced Block Attention Module)**: Combines both channel and spatial attention mechanisms to better capture important features in the input data.
- **CNN Architecture**: A lightweight CNN backbone for efficient training and prediction.
- **Self-Attention**: A mechanism to capture long-range dependencies within the spatial dimensions of the input data.

## Requirements

- Python 3.x
- PyTorch (>= 1.9.0)
- NumPy
- Matplotlib
- tqdm
- einops
- netCDF4

You can install the required dependencies using `pip`:

```bash
pip install torch numpy matplotlib tqdm einops netCDF4
