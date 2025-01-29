"""
Tests for the AdvancedProcessor class.
"""

import os
import pytest
import numpy as np
from datetime import datetime, timedelta
import torch
from shapely.geometry import box
import xarray as xr

from tileformer.data_acquisition.processors import AdvancedProcessor

@pytest.fixture
def processor():
    """Create an AdvancedProcessor instance for testing."""
    return AdvancedProcessor(
        cache_dir="test_cache",
        enable_gpu=False,
        model_dir="test_models"
    )

@pytest.fixture
def test_data():
    """Create test satellite data."""
    # Create synthetic 4-band image (e.g., B, G, R, NIR)
    data = np.random.rand(4, 100, 100).astype(np.float32)
    return data

@pytest.fixture
def test_time_series():
    """Create test time series data."""
    # Create 12 monthly images
    data = []
    dates = []
    
    start_date = datetime(2024, 1, 1)
    for i in range(12):
        data.append(np.random.rand(4, 100, 100).astype(np.float32))
        dates.append(start_date + timedelta(days=30*i))
    
    return data, dates

def test_init(processor):
    """Test processor initialization."""
    assert processor.cache_dir == "test_cache"
    assert processor.device == torch.device("cpu")
    assert processor.model_dir == "test_models"

def test_cloud_detection(processor, test_data):
    """Test cloud detection."""
    # Test statistical method
    mask = processor.detect_clouds(
        test_data,
        method="statistical",
        threshold=0.5
    )
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    
    # Test threshold method
    mask = processor.detect_clouds(
        test_data,
        method="threshold",
        threshold=0.5
    )
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    
    # Test invalid method
    with pytest.raises(ValueError):
        processor.detect_clouds(test_data, method="invalid")

def test_spectral_indices(processor, test_data):
    """Test spectral indices calculation."""
    bands = {
        "BLUE": 0,
        "GREEN": 1,
        "RED": 2,
        "NIR": 3
    }
    
    indices = ["NDVI", "NDWI", "EVI", "SAVI"]
    
    results = processor.calculate_indices(
        test_data,
        bands,
        indices
    )
    
    assert isinstance(results, dict)
    assert all(index in results for index in indices)
    assert all(isinstance(result, np.ndarray) for result in results.values())
    assert all(result.shape == (100, 100) for result in results.values())
    
    # Test invalid index
    with pytest.raises(ValueError):
        processor.calculate_indices(test_data, bands, ["invalid"])

def test_change_detection(processor, test_data):
    """Test change detection."""
    # Create slightly modified data
    data2 = test_data + np.random.normal(0, 0.1, test_data.shape)
    
    # Test difference method
    mask = processor.detect_changes(
        test_data,
        data2,
        method="difference",
        threshold=0.1
    )
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    
    # Test ratio method
    mask = processor.detect_changes(
        test_data,
        data2,
        method="ratio",
        threshold=0.1
    )
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    
    # Test invalid method
    with pytest.raises(ValueError):
        processor.detect_changes(test_data, data2, method="invalid")

def test_super_resolution(processor, test_data):
    """Test super-resolution."""
    # Test bicubic method
    enhanced = processor.super_resolution(
        test_data,
        scale_factor=2,
        method="bicubic"
    )
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == (4, 200, 200)
    assert enhanced.dtype == test_data.dtype
    
    # Test invalid method
    with pytest.raises(ValueError):
        processor.super_resolution(test_data, method="invalid")

def test_pansharpening(processor, test_data):
    """Test pansharpening."""
    # Create panchromatic band
    pan_data = np.mean(test_data[:3], axis=0)
    
    # Test Brovey transform
    sharpened = processor.pansharpen(
        test_data,
        pan_data,
        method="brovey"
    )
    assert isinstance(sharpened, np.ndarray)
    assert sharpened.shape == test_data.shape
    assert sharpened.dtype == test_data.dtype
    
    # Test IHS transform
    sharpened = processor.pansharpen(
        test_data,
        pan_data,
        method="ihs"
    )
    assert isinstance(sharpened, np.ndarray)
    assert sharpened.shape == test_data.shape
    assert sharpened.dtype == test_data.dtype
    
    # Test PCA transform
    sharpened = processor.pansharpen(
        test_data,
        pan_data,
        method="pca"
    )
    assert isinstance(sharpened, np.ndarray)
    assert sharpened.shape == test_data.shape
    assert sharpened.dtype == test_data.dtype
    
    # Test invalid method
    with pytest.raises(ValueError):
        processor.pansharpen(test_data, pan_data, method="invalid")

def test_segmentation(processor, test_data):
    """Test image segmentation."""
    # Test watershed method
    segments = processor.segment_objects(
        test_data,
        method="watershed",
        min_size=100
    )
    assert isinstance(segments, np.ndarray)
    assert segments.shape == (100, 100)
    assert segments.dtype == np.int32
    
    # Test invalid method
    with pytest.raises(ValueError):
        processor.segment_objects(test_data, method="invalid")

def test_time_series_analysis(processor, test_time_series):
    """Test time series analysis."""
    data, dates = test_time_series
    
    # Test linear trend analysis
    results = processor.analyze_time_series(
        data,
        dates,
        method="linear"
    )
    assert isinstance(results, dict)
    assert "trend" in results
    assert isinstance(results["trend"], np.ndarray)
    assert results["trend"].shape == (4, 100, 100)
    
    # Test seasonal decomposition
    results = processor.analyze_time_series(
        data,
        dates,
        method="seasonal"
    )
    assert isinstance(results, dict)
    assert "decomposition" in results
    assert all(f"band_{i}" in results["decomposition"] for i in range(4))
    
    # Test invalid method
    with pytest.raises(ValueError):
        processor.analyze_time_series(data, dates, method="invalid")

def test_model_loading(processor):
    """Test model loading."""
    # Test missing model directory
    processor.model_dir = None
    with pytest.raises(ValueError):
        processor._load_model("test_model")
    
    # Test missing model file
    processor.model_dir = "test_models"
    with pytest.raises(FileNotFoundError):
        processor._load_model("nonexistent_model")

def test_stress(processor):
    """Stress test with large data."""
    # Create large dataset
    large_data = np.random.rand(4, 1000, 1000).astype(np.float32)
    
    # Test cloud detection
    mask = processor.detect_clouds(large_data)
    assert mask.shape == (1000, 1000)
    
    # Test super-resolution
    enhanced = processor.super_resolution(large_data[:, :100, :100])
    assert enhanced.shape == (4, 200, 200)
    
    # Test segmentation
    segments = processor.segment_objects(large_data[:, :100, :100])
    assert segments.shape == (100, 100)

def test_edge_cases(processor):
    """Test edge cases."""
    # Test empty data
    empty_data = np.array([])
    with pytest.raises(ValueError):
        processor.detect_clouds(empty_data)
    
    # Test single pixel
    single_pixel = np.random.rand(4, 1, 1)
    mask = processor.detect_clouds(single_pixel)
    assert mask.shape == (1, 1)
    
    # Test NaN values
    nan_data = np.random.rand(4, 100, 100)
    nan_data[0, 0, 0] = np.nan
    mask = processor.detect_clouds(nan_data)
    assert not np.isnan(mask).any() 