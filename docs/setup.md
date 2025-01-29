# Vortx Setup Guide

This guide will help you set up Vortx for earth observation analysis, with a focus on memory-based processing and advanced analytics.

## Prerequisites

1. **Python Environment**
   - Python 3.9 or higher
   - pip package manager
   - virtualenv or conda (recommended)

2. **System Requirements**
   - 16GB RAM minimum (32GB recommended)
   - NVIDIA GPU with CUDA support (optional, but recommended)
   - 100GB disk space for data storage

3. **API Keys**
   - Google Earth Engine account
   - OpenWeatherMap API key (optional)
   - Space-Track.org account (optional)
   - Sentinel Hub account (optional)

## Installation

### 1. Basic Installation

```bash
# Create and activate virtual environment
python -m venv vortx-env
source vortx-env/bin/activate  # Linux/Mac
# or
.\vortx-env\Scripts\activate  # Windows

# Install from PyPI
pip install vortx

# Install with all extras
pip install vortx[all]
```

### 2. Development Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vortx.git
cd vortx

# Install in development mode
pip install -e ".[dev]"
```

### 3. Docker Installation

```bash
# Pull pre-built image
docker pull vortx/vortx:latest

# Or build from source
docker build -t vortx:latest .
```

## Configuration

### 1. Environment Variables

Create a `.env` file:

```bash
# API Keys
VORTX_OPENWEATHERMAP_KEY=your_key_here
VORTX_SPACETRACK_USERNAME=your_username
VORTX_SPACETRACK_PASSWORD=your_password
VORTX_SENTINEL_HUB_KEY=your_key_here

# Data Paths
VORTX_DATA_DIR=/path/to/data
VORTX_CACHE_DIR=/path/to/cache

# Processing
VORTX_USE_GPU=true
VORTX_NUM_WORKERS=4
VORTX_MEMORY_LIMIT=16GB
```

### 2. Data Directory Structure

```bash
mkdir -p data/{satellite,elevation,climate,memories,cache}
```

### 3. Initialize Earth Engine

```python
import ee
ee.Authenticate()
ee.Initialize()
```

## Basic Usage

### 1. Memory Store Setup

```python
from vortx.core.memory import EarthMemoryStore
from pathlib import Path

# Initialize memory store
memory_store = EarthMemoryStore(
    base_path=Path("./data/memories"),
    index_type="faiss"
)
```

### 2. Data Source Configuration

```python
from vortx.core.data_sources import (
    SatelliteDataSource,
    WeatherDataSource,
    ElevationDataSource
)

# Configure data sources
data_sources = [
    SatelliteDataSource(
        name="sentinel2",
        resolution=10.0,
        bands=["B02", "B03", "B04", "B08"],
        data_path=Path("./data/satellite")
    ),
    WeatherDataSource(
        name="weather",
        resolution=1000.0,
        api_key=os.getenv("VORTX_OPENWEATHERMAP_KEY")
    ),
    ElevationDataSource(
        name="elevation",
        resolution=30.0,
        data_path=Path("./data/elevation")
    )
]
```

### 3. Pipeline Setup

```python
from vortx.core.synthesis import SynthesisPipeline

# Create pipeline
pipeline = SynthesisPipeline(
    data_sources=data_sources,
    memory_store=memory_store
)
```

## Advanced Configuration

### 1. Custom Memory Encoder

```python
from vortx.core.memory import EarthMemoryEncoder
import torch.nn as nn

class CustomEncoder(EarthMemoryEncoder):
    def __init__(self):
        super().__init__(
            input_channels=12,
            embedding_dim=512,
            temporal_window=24,
            use_attention=True
        )
        # Add custom layers
        self.additional_layer = nn.Linear(512, 512)
        
    def forward(self, x):
        embedding, attention_maps = super().forward(x)
        return self.additional_layer(embedding), attention_maps

# Use custom encoder
memory_store = EarthMemoryStore(
    base_path=Path("./data/memories"),
    encoder=CustomEncoder()
)
```

### 2. Custom Data Source

```python
from vortx.core.synthesis import DataSource

class CustomDataSource(DataSource):
    def __init__(self, name: str, resolution: float):
        super().__init__(name, resolution)
        
    def load_data(self, coordinates, timestamp, window_size=(256, 256)):
        # Implementation
        pass
        
    def preprocess(self, data):
        # Implementation
        pass
```

### 3. Performance Tuning

```python
# Cache configuration
import torch
torch.backends.cudnn.benchmark = True

# Memory management
import os
os.environ["VORTX_MEMORY_LIMIT"] = "32GB"

# Parallel processing
from vortx.utils.parallel import set_num_workers
set_num_workers(8)
```

## API Server Setup

### 1. Basic Server

```bash
# Install server dependencies
pip install "vortx[server]"

# Run server
uvicorn vortx.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Production Deployment

```bash
# Using Docker Compose
docker-compose up -d

# Or using Kubernetes
kubectl apply -f k8s/
```

### 3. Load Balancing

```nginx
# nginx.conf
upstream vortx {
    server vortx1:8000;
    server vortx2:8000;
    server vortx3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://vortx;
    }
}
```

## Troubleshooting

### 1. Memory Issues

```python
# Monitor memory usage
from vortx.utils.monitoring import memory_usage

with memory_usage() as mem:
    pipeline.process_location(...)
    print(f"Peak memory: {mem.peak / 1e9:.2f} GB")
```

### 2. GPU Issues

```python
# Check GPU availability
from vortx.utils.gpu import check_gpu

if not check_gpu():
    print("GPU not available, falling back to CPU")
```

### 3. Data Source Issues

```python
# Test data source connectivity
from vortx.utils.testing import test_data_sources

results = test_data_sources(data_sources)
for source, status in results.items():
    print(f"{source}: {'OK' if status else 'Failed'}")
```

## Maintenance

### 1. Cache Cleanup

```python
from vortx.utils.maintenance import cleanup_cache

cleanup_cache(
    cache_dir=Path("./data/cache"),
    max_age_days=30
)
```

### 2. Index Optimization

```python
# Optimize FAISS index
memory_store.optimize_index()
```

### 3. Data Updates

```python
# Update satellite data
from vortx.utils.data import update_satellite_data

update_satellite_data(
    data_path=Path("./data/satellite"),
    start_date="2024-01-01",
    end_date="2024-03-15"
)
```

## Next Steps

1. Check out the [Examples](examples.md) for more use cases
2. Read the [API Documentation](api.md) for detailed reference
3. Join our [Discord Community](https://discord.gg/vortx) for support
4. Contribute to the project on [GitHub](https://github.com/yourusername/vortx) 