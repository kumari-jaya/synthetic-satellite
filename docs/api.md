# TileFormer API Documentation

## Overview

TileFormer provides a powerful REST API for advanced geospatial data processing and machine learning inference. This document details the available endpoints, parameters, and usage examples.

## Base URL

```
https://api.tileformer.io/v2
```

## Authentication

TileFormer uses API keys for authentication. Include your API key in the request header:

```bash
Authorization: Bearer your_api_key
```

## Endpoints

### 1. Tile Endpoint

```
GET /tiles/{z}/{x}/{y}.{format}
```

Serves vector and raster tiles with real-time processing and ML capabilities.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| z | integer | Zoom level |
| x | integer | Tile X coordinate |
| y | integer | Tile Y coordinate |
| format | string | Output format (mvt, pbf, png, jpg, webp) |
| style | string | Style specification (optional) |
| layers | string | Comma-separated layer names (optional) |
| time | string | ISO timestamp for temporal queries (optional) |
| filter | string | Spatial filter specification (optional) |
| transform | string | Transformation pipeline (optional) |
| ml_model | string | ML model name (optional) |
| ml_task | string | ML task type (optional) |
| confidence_threshold | float | Confidence threshold for ML (default: 0.5) |
| prompt | string | Text prompt for generative models (optional) |
| points | string | JSON array of points for SAM (optional) |
| boxes | string | JSON array of boxes for detection (optional) |
| return_type | string | Output type (array/geodataframe) (optional) |
| steps | integer | Number of diffusion steps (default: 50) |
| guidance | float | Guidance scale for diffusion (default: 7.5) |

#### Example Requests

1. Basic Vector Tile
```bash
curl "https://api.tileformer.io/v2/tiles/12/2048/1024.mvt" \
  -H "Authorization: Bearer your_api_key"
```

2. Raster Tile with ML
```bash
curl "https://api.tileformer.io/v2/tiles/15/16384/8192.png?ml_model=sam-vit-huge&ml_task=segmentation" \
  -H "Authorization: Bearer your_api_key"
```

3. Advanced Processing
```bash
curl "https://api.tileformer.io/v2/tiles/14/4096/4096.webp?style=custom&transform=normalize,sharpen&ml_model=sdxl&prompt=enhance satellite imagery" \
  -H "Authorization: Bearer your_api_key"
```

### 2. Batch Processing

```
POST /tiles/batch
```

Process multiple tiles in parallel.

#### Request Body

```json
{
  "tiles": [
    {
      "z": 12,
      "x": 2048,
      "y": 1024,
      "format": "mvt",
      "style": "custom"
    },
    {
      "z": 15,
      "x": 16384,
      "y": 8192,
      "format": "png",
      "ml_model": "sam-vit-huge",
      "ml_task": "segmentation"
    }
  ],
  "concurrent": 4
}
```

### 3. ML Inference

```
POST /ml/infer
```

Direct ML inference on images.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| image | file | Input image file |
| model | string | ML model name |
| task | string | ML task type |
| confidence_threshold | float | Confidence threshold (default: 0.5) |
| prompt | string | Text prompt (optional) |
| points | string | JSON array of points (optional) |
| boxes | string | JSON array of boxes (optional) |
| return_type | string | Output type (optional) |
| steps | integer | Number of steps (optional) |
| guidance | float | Guidance scale (optional) |

#### Example Request

```bash
curl -X POST "https://api.tileformer.io/v2/ml/infer" \
  -H "Authorization: Bearer your_api_key" \
  -F "image=@input.png" \
  -F "model=sam-vit-huge" \
  -F "task=segmentation" \
  -F "confidence_threshold=0.7"
```

## Advanced Features

### 1. Vector Processing

- **Dynamic Simplification**: Automatic topology-preserving simplification
- **Real-time Generalization**: Scale-dependent feature generalization
- **Advanced Filtering**: Complex spatial and attribute queries
- **Custom Styling**: Dynamic styling with expressions

### 2. Raster Processing

- **Band Math**: Real-time band calculations and indices
- **Color Enhancement**: Advanced color correction and balancing
- **Resampling**: Multiple resampling algorithms
- **Filtering**: Customizable filter chains

### 3. ML Capabilities

#### Segmentation
- SAM (Segment Anything Model)
- SegFormer
- DeepLabV3+

#### Object Detection
- YOLOv5
- Faster R-CNN

#### Classification
- ResNet
- EfficientNet

#### Super-resolution
- ESRGAN
- Real-ESRGAN

#### Generative Models
- Stable Diffusion
- ControlNet
- SDXL

#### Vision Language Models
- DeepSeek Vision

### 4. Optimization Features

- **TensorRT Integration**: GPU-optimized inference
- **Custom CUDA Kernels**: Specialized spatial operations
- **Advanced Caching**: Multi-level intelligent caching
- **Parallel Processing**: Distributed computation support

## Error Handling

TileFormer uses standard HTTP status codes and returns detailed error messages:

```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "Invalid ML model specified",
    "details": {
      "parameter": "ml_model",
      "available_models": ["sam-vit-huge", "segformer-b0"]
    }
  }
}
```

## Rate Limiting

- Standard tier: 100 requests/minute
- Professional tier: 1000 requests/minute
- Enterprise tier: Custom limits

## Best Practices

1. **Caching**
   - Use appropriate cache headers
   - Implement client-side caching
   - Consider using CDN

2. **Performance**
   - Request only needed layers
   - Use appropriate zoom levels
   - Batch requests when possible

3. **ML Processing**
   - Set appropriate confidence thresholds
   - Use optimized model variants
   - Consider task requirements

## SDK Support

TileFormer provides official SDKs for:
- Python
- JavaScript
- R
- Java
- Go

## Examples

Visit our [GitHub repository](https://github.com/tileformer/examples) for complete examples and tutorials. 