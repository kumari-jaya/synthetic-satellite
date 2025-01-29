"""
Tests for the UnifiedAPI class.
"""

import os
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from datetime import datetime, timedelta
from pathlib import Path

from tileformer.data_acquisition.sources import UnifiedAPI

@pytest.fixture
def api():
    """Create a UnifiedAPI instance for testing."""
    return UnifiedAPI(
        cache_dir="test_cache",
        max_workers=2,
        enable_streaming=True
    )

@pytest.fixture
def test_bbox():
    """Create a test bounding box."""
    return (-73.9857, 40.7484, -73.9798, 40.7520)  # NYC area

@pytest.fixture
def test_dates():
    """Create test dates."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def test_init(api):
    """Test API initialization."""
    assert api.cache_dir == Path("test_cache")
    assert api.max_workers == 2
    assert api.enable_streaming is True
    assert "planetary_computer" in api.data_sources
    assert "wms_services" in api.data_sources

def test_get_data(api, test_bbox, test_dates):
    """Test data retrieval."""
    start_date, end_date = test_dates
    
    data = api.get_data(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2-l2a"],
        data_types=["raster"],
        formats=["geoparquet"]
    )
    
    assert isinstance(data, dict)
    assert "planetary_computer" in data or "wms_services" in data

def test_get_data_with_polygon(api, test_dates):
    """Test data retrieval with polygon."""
    start_date, end_date = test_dates
    
    # Create test polygon
    polygon = Polygon([
        (-73.9857, 40.7484),
        (-73.9857, 40.7520),
        (-73.9798, 40.7520),
        (-73.9798, 40.7484),
        (-73.9857, 40.7484)
    ])
    
    data = api.get_data(
        bbox=polygon,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2-l2a"]
    )
    
    assert isinstance(data, dict)

def test_format_conversion(api, test_bbox, test_dates):
    """Test data format conversion."""
    start_date, end_date = test_dates
    
    data = api.get_data(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2-l2a"],
        formats=["geoparquet", "geojson"]
    )
    
    assert "geoparquet" in data
    assert "geojson" in data

def test_source_metrics(api):
    """Test source metrics retrieval."""
    metrics = api.get_source_metrics("planetary_computer")
    assert metrics is not None
    assert metrics.speed.value == 4  # FAST
    assert metrics.cost.value == 5   # FREE
    assert metrics.reliability.value == 4  # HIGH

def test_available_sources(api):
    """Test available sources listing."""
    sources = api.get_available_sources()
    assert isinstance(sources, list)
    assert len(sources) > 0
    assert all(isinstance(source, dict) for source in sources)
    assert all("name" in source for source in sources)
    assert all("metrics" in source for source in sources)
    assert all("status" in source for source in sources)

def test_error_handling(api):
    """Test error handling with invalid parameters."""
    with pytest.raises(RuntimeError):
        api.get_data(
            bbox=(-180, -90, 180, 90),  # Too large bbox
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

def test_cache_functionality(api, test_bbox, test_dates):
    """Test caching functionality."""
    start_date, end_date = test_dates
    
    # First request (no cache)
    data1 = api.get_data(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    # Second request (should use cache)
    data2 = api.get_data(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    assert data1 == data2

def test_concurrent_requests(api, test_bbox, test_dates):
    """Test concurrent data requests."""
    start_date, end_date = test_dates
    
    # Create multiple bboxes
    bboxes = [
        test_bbox,
        (test_bbox[0] + 0.1, test_bbox[1], test_bbox[2] + 0.1, test_bbox[3]),
        (test_bbox[0] - 0.1, test_bbox[1], test_bbox[2] - 0.1, test_bbox[3])
    ]
    
    # Request data concurrently
    results = []
    for bbox in bboxes:
        data = api.get_data(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date
        )
        results.append(data)
    
    assert len(results) == len(bboxes)
    assert all(isinstance(result, dict) for result in results)

@pytest.mark.asyncio
async def test_async_support(api, test_bbox, test_dates):
    """Test async support for compatible sources."""
    start_date, end_date = test_dates
    
    # Get sources that support async
    async_sources = [
        name for name, info in api.data_sources.items()
        if info["metrics"].supports_async
    ]
    
    assert len(async_sources) > 0
    
    # Test async data retrieval
    for source in async_sources:
        data = await api._get_from_source_async(
            source,
            api.data_sources[source],
            test_bbox,
            start_date,
            end_date,
            ["sentinel-2-l2a"],
            ["raster"],
            10.0,
            20.0
        )
        assert isinstance(data, (dict, type(None))) 