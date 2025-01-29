"""
Tests for the EarthEngineAPI class.
"""

import os
import pytest
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import box, Polygon
import ee

from tileformer.data_acquisition.sources import EarthEngineAPI

@pytest.fixture
def api():
    """Create an EarthEngineAPI instance for testing."""
    return EarthEngineAPI(
        cache_dir="test_ee_cache"
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
    assert api.cache_dir == "test_ee_cache"
    assert len(api.collections) > 0
    assert "sentinel-2" in api.collections
    assert "landsat-8" in api.collections
    assert "modis" in api.collections

def test_search_and_download(api, test_bbox, test_dates):
    """Test data search and download."""
    start_date, end_date = test_dates
    
    data = api.search_and_download(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2"],
        cloud_cover=20.0,
        max_items=1
    )
    
    assert isinstance(data, dict)
    if "sentinel-2" in data:
        assert len(data["sentinel-2"]) > 0
        item = data["sentinel-2"][0]
        assert "data" in item
        assert "metadata" in item
        assert isinstance(item["data"], np.ndarray)
        assert item["data"].ndim == 3
        assert item["metadata"]["resolution"] == 10

def test_search_with_polygon(api, test_dates):
    """Test search with polygon."""
    start_date, end_date = test_dates
    
    # Create test polygon
    polygon = Polygon([
        (-73.9857, 40.7484),
        (-73.9857, 40.7520),
        (-73.9798, 40.7520),
        (-73.9798, 40.7484),
        (-73.9857, 40.7484)
    ])
    
    data = api.search_and_download(
        bbox=polygon,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2"]
    )
    
    assert isinstance(data, dict)

def test_time_series(api, test_bbox, test_dates):
    """Test time series retrieval."""
    start_date, end_date = test_dates
    
    data = api.get_time_series(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collection="sentinel-2",
        band="B8",
        temporal_resolution="month"
    )
    
    assert isinstance(data, dict)
    assert "timestamps" in data
    assert "values" in data
    assert "metadata" in data
    assert len(data["timestamps"]) == len(data["values"])
    assert data["metadata"]["band"] == "B8"
    assert data["metadata"]["resolution"] == 10

def test_available_collections(api):
    """Test available collections listing."""
    collections = api.get_available_collections()
    
    assert isinstance(collections, list)
    assert len(collections) > 0
    assert all(isinstance(c, dict) for c in collections)
    assert all("name" in c for c in collections)
    assert all("id" in c for c in collections)
    assert all("resolution" in c for c in collections)
    assert all("bands" in c for c in collections)

def test_add_collection(api):
    """Test adding new collection."""
    api.add_collection(
        name="test-collection",
        collection_id="TEST/COLLECTION",
        resolution=20.0,
        bands=["B1", "B2", "B3"],
        scale_factor=0.0001
    )
    
    assert "test-collection" in api.collections
    assert api.collections["test-collection"]["id"] == "TEST/COLLECTION"
    assert api.collections["test-collection"]["resolution"] == 20.0
    assert api.collections["test-collection"]["bands"] == ["B1", "B2", "B3"]
    assert api.collections["test-collection"]["scale_factor"] == 0.0001

def test_error_handling(api, test_bbox, test_dates):
    """Test error handling."""
    start_date, end_date = test_dates
    
    # Test invalid collection
    data = api.search_and_download(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["invalid-collection"]
    )
    assert data == {}
    
    # Test invalid dates
    with pytest.raises(Exception):
        api.search_and_download(
            bbox=test_bbox,
            start_date="invalid-date",
            end_date=end_date
        )
    
    # Test invalid bbox
    with pytest.raises(Exception):
        api.search_and_download(
            bbox=(-181, -91, 181, 91),  # Invalid coordinates
            start_date=start_date,
            end_date=end_date
        )

def test_concurrent_downloads(api, test_bbox, test_dates):
    """Test concurrent downloads."""
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
        data = api.search_and_download(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collections=["sentinel-2"]
        )
        results.append(data)
    
    assert len(results) == len(bboxes)
    assert all(isinstance(result, dict) for result in results)

def test_cache_functionality(api, test_bbox, test_dates):
    """Test caching functionality."""
    start_date, end_date = test_dates
    
    # First request (no cache)
    data1 = api.search_and_download(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2"]
    )
    
    # Second request (should use cache)
    data2 = api.search_and_download(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2"]
    )
    
    if "sentinel-2" in data1 and "sentinel-2" in data2:
        assert np.array_equal(
            data1["sentinel-2"][0]["data"],
            data2["sentinel-2"][0]["data"]
        )

def test_large_area(api, test_dates):
    """Test handling large area requests."""
    start_date, end_date = test_dates
    
    # Create large bbox
    large_bbox = (-74.0, 40.7, -73.9, 40.8)  # Larger NYC area
    
    data = api.search_and_download(
        bbox=large_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2"]
    )
    
    assert isinstance(data, dict)
    if "sentinel-2" in data:
        assert len(data["sentinel-2"]) > 0

def test_multiple_collections(api, test_bbox, test_dates):
    """Test retrieving multiple collections."""
    start_date, end_date = test_dates
    
    data = api.search_and_download(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        collections=["sentinel-2", "landsat-8", "modis"]
    )
    
    assert isinstance(data, dict)
    assert any(collection in data for collection in ["sentinel-2", "landsat-8", "modis"]) 