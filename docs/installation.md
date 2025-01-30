# Installation Guide

## Quick Start

### From Source (Recommended)
```bash
# Clone the repository
git clone https://github.com/vortx-ai/vortx.git
cd vortx

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Optional: Install development dependencies
pip install -e .[dev]
```

### Using pip (Coming Soon)
```bash
# Note: Package not yet available on PyPI
# Expected release: Q2 2024

# Basic installation
pip install vortx  # Coming soon

# Install with all optional dependencies
pip install vortx[all]  # Coming soon

# Install specific feature sets
pip install vortx[gpu]     # GPU acceleration
pip install vortx[ml]      # Machine learning features
pip install vortx[viz]     # Visualization tools
```

### Using Docker (Coming Soon)
```bash
# Note: Images not yet available on Docker Hub
# Expected release: Q2 2024

# Pull the latest image
docker pull vortx/vortx:latest  # Coming soon

# Run with GPU support
docker run --gpus all -p 8000:8000 vortx/vortx:latest

# Run with mounted data directory
docker run -v /path/to/data:/data -p 8000:8000 vortx/vortx:latest
```

## System Requirements

### Minimum Requirements
- Python 3.9+
- 8GB RAM
- 4 CPU cores
- 10GB disk space

### Recommended Requirements
- Python 3.9+
- 32GB RAM
- 8+ CPU cores
- NVIDIA GPU with 8GB+ VRAM
- 50GB SSD storage

### Optional Requirements
- CUDA 11.x or later (for GPU acceleration)
- Docker 20.10 or later (coming soon)
- Node.js 16+ (for web interface, coming soon)

## Feature-specific Dependencies

### GPU Acceleration
```bash
# Install CUDA toolkit first
# For Ubuntu:
sudo apt-get update
sudo apt-get install -y nvidia-driver-525 nvidia-cuda-toolkit

# Install GPU dependencies
pip install -r requirements-gpu.txt
```

### Machine Learning
```bash
# Install ML dependencies
pip install -r requirements-ml.txt

# Optional: Install specific model weights (coming soon)
# vo download-models --model deepseek-vl
```

### Visualization Tools
```bash
# Install visualization dependencies
pip install -r requirements-viz.txt

# Required system packages (Ubuntu)
sudo apt-get install -y libgl1-mesa-glx
```

## Environment Setup

### Environment Variables
```bash
# Core settings
export VORTX_HOME=/path/to/vortx/data
export VORTX_CACHE_DIR=/path/to/cache
export VORTX_CONFIG=/path/to/config.yaml

# API keys (if using cloud services)
export VORTX_API_KEY=your_api_key
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### Configuration File
```yaml
# config.yaml
storage:
  type: local  # or s3, gcs
  path: /path/to/data
  cache_size: 10GB

compute:
  device: cuda  # or cpu
  num_workers: 4
  batch_size: 32

api:
  host: 0.0.0.0
  port: 8000
  debug: false
```

## Development Setup

### Setting up for Development
```bash
# Clone the repository
git clone https://github.com/vortx-ai/vortx.git
cd vortx

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run style checks
black vortx/
isort vortx/
flake8 vortx/
```

### Building Documentation
```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Memory Issues**
```bash
# Adjust memory limits in config.yaml
memory:
  cache_size: 4GB
  max_batch_size: 16
```

3. **Import Errors**
```bash
# Verify installation
python -c "import vortx; print(vortx.__version__)"

# Check Python path
python -c "import sys; print(sys.path)"
```

### Getting Help

- **Documentation**: Visit [https://vortx.ai/docs](https://vortx.ai/docs)
- **GitHub Issues**: Report bugs at [https://github.com/vortx-ai/vortx/issues](https://github.com/vortx-ai/vortx/issues)
- **Discord Community**: Join our [Discord server](https://discord.gg/vortx)
- **Email Support**: Contact support@vortx.ai

## Upgrading

### Version Upgrades
```bash
# Currently only available through git
git pull origin main
pip install -e .
```

### Migration Guide
- See [MIGRATION.md](MIGRATION.md) for version-specific upgrade notes
- Back up your configuration before upgrading
- Test in a staging environment first

## Security Considerations

### API Keys
- Store API keys in environment variables
- Use `.env` files for local development
- Never commit API keys to version control

### Network Security
- Configure firewall rules
- Use HTTPS for production
- Set appropriate CORS policies

### Data Privacy
- Enable encryption at rest
- Configure access controls
- Follow data retention policies 