# TileFormer Deployment Guide

## Overview

This guide covers deploying TileFormer in various environments, from development to production. We'll cover:
- Local development setup
- Docker containerization
- Cloud deployment
- Performance optimization
- Monitoring and maintenance

## Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 8GB+ GPU memory (recommended)
- 100GB+ storage

## 1. Local Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Configuration

Create a `.env` file:

```env
TILEFORMER_ENV=development
CACHE_DIR=/path/to/cache
MODEL_DIR=/path/to/models
API_KEY=your_api_key
GPU_MEMORY_FRACTION=0.8
MAX_BATCH_SIZE=4
```

### Running Locally

```bash
# Start the server
uvicorn tileformer.api.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Run specific test
pytest tests/test_ml_processor.py -v
```

## 2. Docker Deployment

### Building the Image

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt -r requirements-prod.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV TILEFORMER_ENV=production

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "tileformer.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  tileformer:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./cache:/app/cache
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_DIR=/app/models
      - CACHE_DIR=/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:6.2
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Running with Docker

```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale tileformer=3
```

## 3. Cloud Deployment

### AWS Deployment

1. **ECS Setup**

```bash
# Configure AWS CLI
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name tileformer-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

2. **Task Definition**

```json
{
  "family": "tileformer",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "tileformer",
      "image": "your-repo/tileformer:latest",
      "memory": 8192,
      "cpu": 2048,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "TILEFORMER_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tileformer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192"
}
```

### Kubernetes Deployment

1. **Deployment Configuration**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tileformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tileformer
  template:
    metadata:
      labels:
        app: tileformer
    spec:
      containers:
      - name: tileformer
        image: your-repo/tileformer:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: TILEFORMER_ENV
          value: "production"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: cache
        persistentVolumeClaim:
          claimName: cache-pvc
```

2. **Service Configuration**

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tileformer
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: tileformer
```

## 4. Performance Optimization

### Model Optimization

1. **TensorRT Integration**
```python
# Convert models to TensorRT
from tileformer.utils.optimization import convert_to_tensorrt

model_path = "models/sam-vit-huge"
optimized_path = convert_to_tensorrt(model_path)
```

2. **Quantization**
```python
# Quantize models
from tileformer.utils.optimization import quantize_model

model_path = "models/segformer-b0"
quantized_path = quantize_model(model_path, quantization="int8")
```

### Caching Strategy

1. **Redis Configuration**
```python
# Configure Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'max_memory': '2gb',
    'eviction_policy': 'allkeys-lru'
}
```

2. **Cache Warmup**
```python
# Warm up cache for common tiles
from tileformer.utils.cache import warm_cache

warm_cache(
    zoom_levels=[12, 13, 14],
    bbox=[-122.4, 37.7, -122.3, 37.8]
)
```

## 5. Monitoring and Maintenance

### Prometheus Metrics

```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

REQUESTS = Counter('tileformer_requests_total', 'Total requests')
LATENCY = Histogram('tileformer_request_latency_seconds', 'Request latency')
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "TileFormer Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(tileformer_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Latency",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(tileformer_request_latency_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
        "gpu_memory": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }
```

## 6. Security

### API Authentication

```python
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

@app.get("/secure-endpoint")
async def secure_endpoint(api_key: str = Depends(API_KEY_HEADER)):
    if not verify_api_key(api_key):
        raise HTTPException(status_code=403)
    return {"message": "Authenticated"}
```

### Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/rate-limited")
@limiter.limit("100/minute")
async def rate_limited(request: Request):
    return {"message": "Rate limited endpoint"}
```

## 7. Troubleshooting

### Common Issues

1. **GPU Memory Issues**
```bash
# Check GPU usage
nvidia-smi

# Clear GPU cache
torch.cuda.empty_cache()
```

2. **Performance Issues**
```bash
# Profile code
python -m cProfile -o profile.stats your_script.py
snakeviz profile.stats
```

3. **Memory Leaks**
```bash
# Monitor memory
from memory_profiler import profile

@profile
def memory_intensive_function():
    pass
```

## 8. Maintenance

### Backup Strategy

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backup_$DATE.tar.gz \
    models/ \
    cache/ \
    config/
aws s3 cp backup_$DATE.tar.gz s3://your-bucket/backups/
```

### Update Procedure

```bash
# Update script
#!/bin/bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Restart services
supervisorctl restart tileformer
```

## 9. Scaling

### Horizontal Scaling

```bash
# Scale with Docker Compose
docker-compose up -d --scale tileformer=5

# Scale with Kubernetes
kubectl scale deployment tileformer --replicas=5
```

### Vertical Scaling

- Increase instance size
- Add more GPU memory
- Optimize model loading

## 10. Best Practices

1. **Production Checklist**
   - [ ] Security hardening
   - [ ] Monitoring setup
   - [ ] Backup strategy
   - [ ] Rate limiting
   - [ ] Error handling
   - [ ] Documentation
   - [ ] Performance optimization
   - [ ] Load testing

2. **Performance Tips**
   - Use batch processing
   - Implement caching
   - Optimize model loading
   - Use async processing
   - Monitor resource usage

3. **Security Tips**
   - Regular updates
   - API key rotation
   - Input validation
   - Rate limiting
   - Access logging
``` 