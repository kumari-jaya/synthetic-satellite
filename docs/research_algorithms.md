# Research Algorithms and Satellite Pipelines

## Overview

TileFormer implements state-of-the-art research algorithms for satellite image processing and analysis. This document details the available algorithms, their configurations, and example use cases.

## Atmospheric Correction Algorithms

### 1. 6S (Second Simulation of Satellite Signal)
```python
result = tf.atmospheric_correction(
    method="6s",
    parameters={
        "aerosol_profile": "continental",
        "aerosol_optical_depth": 0.2,
        "water_vapor": 2.0,
        "ozone": 0.3,
        "altitude": 0.2,
        "wavelength": [0.45, 0.52, 0.64, 0.85]
    }
)
```

### 2. MODTRAN
```python
result = tf.atmospheric_correction(
    method="modtran",
    parameters={
        "atmosphere_model": "midlatitude_summer",
        "aerosol_model": "rural",
        "visibility": 23.0,
        "ground_altitude": 0.2,
        "sensor_altitude": 705.0
    }
)
```

### 3. Sen2Cor
```python
result = tf.atmospheric_correction(
    method="sen2cor",
    parameters={
        "aerosol_type": "rural",
        "mid_latitude": "summer",
        "ozone_content": 330,
        "water_vapor": "auto",
        "cirrus_correction": True
    }
)
```

## Super-Resolution Algorithms

### 1. SRCNN for Satellite
```python
result = tf.super_resolution(
    method="srcnn_satellite",
    parameters={
        "scale_factor": 4,
        "model_depth": 9,
        "residual_learning": True,
        "spectral_bands": ["B2", "B3", "B4", "B8"]
    }
)
```

### 2. DeepSUM
```python
result = tf.super_resolution(
    method="deepsum",
    parameters={
        "temporal_window": 7,
        "target_resolution": 5,
        "fusion_mode": "attention",
        "registration_net": "siamese"
    }
)
```

### 3. HighRes-net
```python
result = tf.super_resolution(
    method="highres_net",
    parameters={
        "num_frames": 5,
        "scale": 4,
        "fusion_mode": "recursive",
        "attention_type": "spatial"
    }
)
```

## Cloud Detection Algorithms

### 1. Fmask 4.0
```python
result = tf.cloud_detection(
    method="fmask4",
    parameters={
        "probability_threshold": 0.5,
        "cloud_buffer": 50,
        "shadow_buffer": 30,
        "snow_threshold": 0.4
    }
)
```

### 2. s2cloudless
```python
result = tf.cloud_detection(
    method="s2cloudless",
    parameters={
        "threshold": 0.4,
        "all_bands": True,
        "average_over": 4,
        "dilation_size": 2
    }
)
```

## Change Detection Algorithms

### 1. COLD
```python
result = tf.change_detection(
    method="cold",
    parameters={
        "temporal_window": 365,
        "confidence_threshold": 0.95,
        "minimum_observations": 6,
        "harmonic_order": 3
    }
)
```

### 2. CCDC
```python
result = tf.change_detection(
    method="ccdc",
    parameters={
        "num_coefficients": 6,
        "chi_square_threshold": 0.99,
        "min_observations": 12,
        "spectral_indices": ["NDVI", "NBR", "EVI"]
    }
)
```

## Standard Processing Pipelines

### 1. Optical Pipeline
```python
pipeline = tf.Pipeline()
pipeline.add_steps([
    tf.LoadData(source="sentinel-2"),
    tf.AtmosphericCorrection(method="sen2cor"),
    tf.CloudMasking(method="s2cloudless"),
    tf.SpectralIndices(indices=["ndvi", "evi", "ndwi"]),
    tf.SuperResolution(method="deepsum"),
    tf.Export(format="COG")
])
```

### 2. SAR Pipeline
```python
pipeline = tf.Pipeline()
pipeline.add_steps([
    tf.LoadData(source="sentinel-1"),
    tf.Calibration(output="sigma0"),
    tf.SpeckleFilter(method="refined_lee"),
    tf.TerrainCorrection(dem="srtm"),
    tf.PolarimetricDecomposition(method="cloude_pottier"),
    tf.Export(format="GeoTIFF")
])
```

### 3. Time Series Pipeline
```python
pipeline = tf.Pipeline()
pipeline.add_steps([
    tf.LoadTimeSeries(
        source="landsat-8",
        start_date="2020-01-01",
        end_date="2024-01-01"
    ),
    tf.Harmonization(reference="sentinel-2"),
    tf.GapFilling(method="whittaker"),
    tf.TrendAnalysis(methods=["mann_kendall", "theil_sen"]),
    tf.ChangeDetection(method="ccdc"),
    tf.Export(format="NetCDF")
])
```

## Advanced Research Applications

### 1. Forest Monitoring
```python
workflow = tf.create_workflow()
workflow.add_steps([
    tf.load_sentinel2(date_range="2024-01"),
    tf.atmospheric_correction("sen2cor"),
    tf.cloud_masking("s2cloudless"),
    tf.calculate_indices(["ndvi", "nbr"]),
    tf.segment_forest("deepforest"),
    tf.estimate_biomass("random_forest"),
    tf.detect_degradation("timescan")
])
```

### 2. Urban Analysis
```python
workflow = tf.create_workflow()
workflow.add_steps([
    tf.load_sentinel2(date_range="2024-01"),
    tf.super_resolution("srcnn", scale=4),
    tf.calculate_indices(["ndbi", "ui"]),
    tf.extract_buildings("mask_rcnn"),
    tf.analyze_urban_growth("deep_change"),
    tf.calculate_metrics(["density", "sprawl"])
])
```

### 3. Agricultural Monitoring
```python
workflow = tf.create_workflow()
workflow.add_steps([
    tf.load_data(sources=["sentinel-2", "sentinel-1"]),
    tf.fusion("optical_sar"),
    tf.calculate_indices(["ndvi", "savi", "rei"]),
    tf.classify_crops("transformer"),
    tf.estimate_yield("lstm"),
    tf.detect_anomalies("isolation_forest")
])
```

## Quality Assessment

### 1. Validation Metrics
- Spatial accuracy assessment
- Temporal consistency check
- Spectral quality indicators
- Classification accuracy metrics

### 2. Uncertainty Estimation
- Error propagation analysis
- Confidence intervals
- Ensemble predictions
- Cross-validation

### 3. Performance Benchmarks
- Processing time
- Memory usage
- Scaling efficiency
- Accuracy vs. speed tradeoffs 