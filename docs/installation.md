# Installation Guide

This guide will help you install Vortx and its dependencies. Vortx supports multiple installation methods depending on your needs.

## Quick Installation

The simplest way to install Vortx is via pip:

```bash
pip install vortx
```

For development installation with all optional dependencies:

```bash
pip install vortx[all]
```

## Requirements

### System Requirements

- Python 3.9 or higher
- CUDA 11.8 or higher (for GPU support)
- At least 8GB RAM (16GB recommended)
- 20GB free disk space

### Core Dependencies

Vortx relies on several key libraries:

- PyTorch (≥2.0.0) for deep learning
- Rasterio (≥1.3.0) for geospatial data handling
- GeoPandas (≥0.12.0) for vector data processing
- Ray (≥2.3.0) and Dask (≥2023.3.0) for distributed computing

## Installation Methods

### 1. Using pip (Recommended)

```bash
# Basic installation
pip install vortx

# With all optional dependencies
pip install vortx[all]

# With specific feature sets
pip install vortx[gpu]  # GPU support
pip install vortx[ml]   # Machine learning features
pip install vortx[viz]  # Visualization tools
pip install vortx[dev]  # Development tools
```

### 2. From Source

```bash
git clone https://github.com/vortx-ai/vortx.git
cd vortx
pip install -e .
```

### 3. Using Docker

```bash
# Pull the latest image
docker pull vortx/vortx:latest

# Run with GPU support
docker run --gpus all -it vortx/vortx:latest
```

## Optional Dependencies

Vortx has several optional dependency groups:

### GPU Acceleration
- CUDA Toolkit ≥11.8
- cupy-cuda11x ≥12.0.0
- onnxruntime-gpu ≥1.15.0
- TensorRT ≥8.6.0

### Machine Learning
- torchvision ≥0.15.0
- transformers ≥4.30.0
- timm ≥0.9.0
- segmentation-models-pytorch ≥0.3.0

### Synthetic Data Generation
- noise ≥1.2.2
- perlin-noise ≥1.12
- trimesh ≥4.0.0
- pyrender ≥0.1.45

### Monitoring and Metrics
- prometheus-client ≥0.16.0
- wandb ≥0.15.0
- mlflow ≥2.7.0

## Environment Setup

### 1. Using conda

```bash
# Create a new conda environment
conda create -n vortx python=3.9
conda activate vortx

# Install Vortx
pip install vortx[all]
```

### 2. Using venv

```bash
# Create a new virtual environment
python -m venv vortx-env
source vortx-env/bin/activate  # Linux/Mac
vortx-env\Scripts\activate     # Windows

# Install Vortx
pip install vortx[all]
```

## Command Line Interface

Vortx provides a convenient command-line interface through the `vo` command:

```bash
# Get help
vo --help

# Process data
vo process input.tif output.tif --model super-res

# Start server
vo serve --port 8000

# Run pipeline
vo pipeline config.yaml
```

## Troubleshooting

### Common Issues

1. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Memory Issues
```bash
# Set memory limits for GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

3. Dependency Conflicts
```bash
# Clean installation
pip uninstall vortx
pip cache purge
pip install vortx --no-cache-dir
```

### Getting Help

If you encounter any issues:

1. Check our [FAQ](https://vortx.ai/docs/faq)
2. Search existing [GitHub Issues](https://github.com/vortx-ai/vortx/issues)
3. Join our [Discord Community](https://discord.gg/vortx)
4. Contact support at support@vortx.ai

## Next Steps

- Read the [Quick Start Guide](./quickstart.md)
- Explore [Examples](./examples/index.md)
- Check out our [API Reference](./api/index.md) 