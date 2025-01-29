"""
Tests for the COGSTACAPI class.
"""

import os
import pytest
import numpy as np
from shapely.geometry import box, Polygon
from datetime import datetime, timedelta
from pathlib import Path
import rasterio
import dask.array as da

from tileformer.data_acquisition.sources import COGSTACAPI

@pytest.fixture
def api():
    """Create a COGSTACAPI instance for testing."""
    return COGSTACAPI(
        cache_dir="test_cog_cache",
        max_workers=2,
        enable_streaming=True,
        chunk_size=512
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

@pytest.fixture
def test_cog_url():
    """Create a test COG URL."""
    return "https://planetarycomputer.microsoft.com/api/data/item/sentinel-2-l2a/S2A_MSIL2A_20220101T123456_N0000_R123_T12ABC_20220101T123456.SAFE/B02.tif"

def test_init(api):
    """Test API initialization."""
    assert api.cache_dir == Path("test_cog_cache")
    assert api.max_workers == 2
    assert api.enable_streaming is True
    assert api.chunk_size == 512
    assert len(api.catalogs) > 0
    assert len(api.clients) > 0

def test_search_collections(api, test_bbox, test_dates):
    """Test collection search."""
    start_date, end_date = test_dates
    
    collections = api.search_collections(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(collections, list)
    if collections:
        assert all(isinstance(c, dict) for c in collections)
        assert all("id" in c for c in collections)
        assert all("extent" in c for c in collections)

def test_search_collections_with_polygon(api, test_dates):
    """Test collection search with polygon."""
    start_date, end_date = test_dates
    
    # Create test polygon
    polygon = Polygon([
        (-73.9857, 40.7484),
        (-73.9857, 40.7520),
        (-73.9798, 40.7520),
        (-73.9798, 40.7484),
        (-73.9857, 40.7484)
    ])
    
    collections = api.search_collections(
        bbox=polygon,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(collections, list)

def test_get_cog_data(api, test_bbox, test_cog_url):
    """Test COG data retrieval."""
    data = api.get_cog_data(
        url=test_cog_url,
        bbox=test_bbox,
        bands=[1],
        resolution=10.0,
        masked=True
    )
    
    assert isinstance(data, dict)
    assert "data" in data
    assert "metadata" in data
    assert isinstance(data["data"], (np.ndarray, da.Array))
    assert isinstance(data["metadata"], dict)

def test_streaming_cog_data(api, test_bbox, test_cog_url):
    """Test streaming COG data."""
    data = api.get_cog_data(
        url=test_cog_url,
        bbox=test_bbox,
        resolution=10.0,
        masked=True
    )
    
    assert isinstance(data["data"], (np.ndarray, da.Array))
    if isinstance(data["data"], da.Array):
        # Check chunk size
        chunks = data["data"].chunks
        assert all(c <= api.chunk_size for c in chunks[1:])

def test_bbox_validation(api):
    """Test bounding box validation."""
    collection_bbox = (-74.0, 40.7, -73.9, 40.8)
    
    # Valid bbox
    valid_bbox = (-73.98, 40.74, -73.97, 40.75)
    assert api._bbox_intersects(valid_bbox, collection_bbox)
    
    # Invalid bbox
    invalid_bbox = (-74.1, 40.6, -74.05, 40.65)
    assert not api._bbox_intersects(invalid_bbox, collection_bbox)

def test_date_validation(api):
    """Test date validation."""
    # Valid dates
    assert api._date_in_range(
        "2024-01-01",
        "2024-01-31",
        "2023-01-01",
        "2024-12-31"
    )
    
    # Invalid dates
    assert not api._date_in_range(
        "2024-01-01",
        "2024-01-31",
        "2024-02-01",
        "2024-12-31"
    )

def test_error_handling(api, test_bbox):
    """Test error handling."""
    # Invalid URL
    with pytest.raises(Exception):
        api.get_cog_data(
            url="invalid_url",
            bbox=test_bbox
        )
    
    # Invalid bbox
    with pytest.raises(Exception):
        api.get_cog_data(
            url=test_cog_url,
            bbox=(-181, -91, 181, 91)  # Invalid coordinates
        )

def test_chunk_resizing(api):
    """Test chunk resizing."""
    # Create test chunk
    chunk = np.random.rand(3, 100, 100)
    
    # Test upscaling
    upscaled = api._resize_chunk(chunk, 2.0)
    assert upscaled.shape == (3, 200, 200)
    
    # Test downscaling
    downscaled = api._resize_chunk(chunk, 0.5)
    assert downscaled.shape == (3, 50, 50)

def test_collection_extent_check(api, test_bbox, test_dates):
    """Test collection extent checking."""
    start_date, end_date = test_dates
    
    # Get a collection
    collections = api.search_collections(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date
    )
    
    if collections:
        collection = collections[0]
        # Valid extent
        assert api._check_collection_extent(
            collection,
            test_bbox,
            start_date,
            end_date
        )
        
        # Invalid spatial extent
        invalid_bbox = (-180, -90, -170, -80)
        assert not api._check_collection_extent(
            collection,
            invalid_bbox,
            start_date,
            end_date
        )
        
        # Invalid temporal extent
        invalid_dates = (
            (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=395)).strftime("%Y-%m-%d")
        )
        assert not api._check_collection_extent(
            collection,
            test_bbox,
            *invalid_dates
        ) 