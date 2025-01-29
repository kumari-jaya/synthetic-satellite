"""
Tests for the MobileMetaverseAPI class.
"""

import os
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from pathlib import Path
from PIL import Image

from tileformer.data_acquisition.sources import MobileMetaverseAPI

@pytest.fixture
def api():
    """Create a MobileMetaverseAPI instance for testing."""
    return MobileMetaverseAPI(
        cache_dir="test_mobile_cache",
        max_texture_size=1024,
        enable_compression=True
    )

@pytest.fixture
def test_vector_data():
    """Create test vector data."""
    # Create simple building polygons
    polygons = [
        box(0, 0, 10, 10),
        box(20, 20, 30, 30)
    ]
    
    data = {
        "geometry": polygons,
        "height": [20.0, 30.0],
        "type": ["residential", "commercial"]
    }
    
    return gpd.GeoDataFrame(data, crs="EPSG:4326")

@pytest.fixture
def test_raster_data():
    """Create test raster data."""
    return np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)

@pytest.fixture
def test_elevation_data():
    """Create test elevation data."""
    return np.random.normal(100, 10, (100, 100))

def test_init(api):
    """Test API initialization."""
    assert api.cache_dir == Path("test_mobile_cache")
    assert api.max_texture_size == 1024
    assert api.enable_compression is True
    assert "3d" in api.formats
    assert "ar" in api.formats
    assert "vr" in api.formats
    assert "terrain" in api.formats

def test_convert_to_3d(api, test_vector_data):
    """Test 3D conversion."""
    model = api.convert_to_3d(
        vector_data=test_vector_data,
        format="glb",
        attributes=["height", "type"],
        height_scale=1.0
    )
    
    assert isinstance(model, dict)
    assert "model_path" in model
    assert "metadata" in model
    assert os.path.exists(model["model_path"])
    assert model["metadata"]["feature_count"] == len(test_vector_data)

def test_convert_to_3d_with_texture(api, test_vector_data, test_raster_data):
    """Test 3D conversion with texture."""
    model = api.convert_to_3d(
        vector_data=test_vector_data,
        raster_data=test_raster_data,
        format="glb",
        attributes=["height"]
    )
    
    assert isinstance(model, dict)
    assert os.path.exists(model["model_path"])

def test_create_ar_scene(api, test_vector_data):
    """Test AR scene creation."""
    # First create a model
    model = api.convert_to_3d(
        vector_data=test_vector_data,
        format="glb"
    )
    
    # Create AR scene
    models = [
        {
            "model_path": model["model_path"],
            "position": [0, 0, 0],
            "rotation": [0, 0, 0]
        }
    ]
    
    scene = api.create_ar_scene(
        models=models,
        format="usdz",
        scene_scale=1.0,
        include_anchors=True
    )
    
    assert isinstance(scene, dict)
    assert "scene_path" in scene
    assert "metadata" in scene
    assert os.path.exists(scene["scene_path"])
    assert scene["metadata"]["model_count"] == len(models)

def test_create_terrain_model(api, test_elevation_data):
    """Test terrain model creation."""
    bbox = (0, 0, 1, 1)
    
    terrain = api.create_terrain_model(
        elevation_data=test_elevation_data,
        bbox=bbox,
        resolution=10.0,
        format="terrain",
        vertical_exaggeration=2.0
    )
    
    assert isinstance(terrain, dict)
    assert "model" in terrain
    assert "metadata" in terrain
    assert os.path.exists(terrain["model"]["path"])
    assert terrain["metadata"]["resolution"] == 10.0
    assert terrain["metadata"]["vertical_exaggeration"] == 2.0

def test_optimize_for_mobile(api, test_vector_data):
    """Test mobile optimization."""
    # Create initial model
    model = api.convert_to_3d(
        vector_data=test_vector_data,
        format="glb"
    )
    
    # Optimize model
    optimized = api.optimize_for_mobile(
        model_data=model,
        target_size="medium",
        preserve_attributes=["height"]
    )
    
    assert isinstance(optimized, dict)
    assert "model_path" in optimized
    assert "metadata" in optimized
    assert os.path.exists(optimized["model_path"])
    assert "optimization" in optimized["metadata"]
    assert optimized["metadata"]["optimization"]["target_size"] == "medium"

def test_create_texture(api, test_raster_data):
    """Test texture creation."""
    texture = api._create_texture(test_raster_data)
    assert isinstance(texture, Image.Image)
    assert texture.size == (test_raster_data.shape[2], test_raster_data.shape[1])

def test_process_elevation(api, test_elevation_data):
    """Test elevation data processing."""
    processed = api._process_elevation(
        test_elevation_data,
        vertical_exaggeration=2.0
    )
    
    assert isinstance(processed, np.ndarray)
    assert processed.shape == test_elevation_data.shape
    assert np.all(processed == test_elevation_data * 2.0)

def test_optimization_params(api):
    """Test optimization parameters."""
    params = api._get_optimization_params("medium")
    assert isinstance(params, dict)
    assert "target_faces" in params
    assert "max_texture_size" in params
    
    with pytest.raises(ValueError):
        api._get_optimization_params("invalid_size")

def test_error_handling(api):
    """Test error handling."""
    with pytest.raises(ValueError):
        api.convert_to_3d(
            vector_data=gpd.GeoDataFrame(),  # Empty dataframe
            format="glb"
        )
    
    with pytest.raises(ValueError):
        api.convert_to_3d(
            vector_data=test_vector_data,
            format="invalid_format"
        )

def test_texture_optimization(api):
    """Test texture optimization."""
    # Create large texture
    large_texture = Image.new("RGB", (4096, 4096))
    
    # Optimize texture
    optimized = api._optimize_texture(
        large_texture,
        max_size=1024
    )
    
    assert isinstance(optimized, Image.Image)
    assert max(optimized.size) <= 1024 