"""Tests for the synthetic data generation module."""

import pytest
import numpy as np
from vortx.synthetic import (
    SyntheticDataGenerator,
    TerrainParams,
    LandCoverParams,
    AtmosphericParams
)
from vortx.utils.exceptions import SyntheticDataError

@pytest.fixture
def generator():
    """Create a synthetic data generator for testing."""
    return SyntheticDataGenerator(
        output_size=(256, 256),
        resolution=10.0,
        num_bands=4,
        device="cpu"
    )

def test_terrain_generation(generator):
    """Test terrain generation with default parameters."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    terrain = generator.generate_terrain(bbox)
    
    assert isinstance(terrain, dict)
    assert "data" in terrain
    assert "transform" in terrain
    assert "crs" in terrain
    assert terrain["data"].shape == (256, 256)
    assert not np.isnan(terrain["data"]).any()

def test_terrain_generation_with_params(generator):
    """Test terrain generation with custom parameters."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    params = TerrainParams(
        elevation_range=(-100, 4000),
        roughness=0.6,
        persistence=0.5,
        lacunarity=2.0,
        octaves=8,
        seed=42
    )
    
    terrain = generator.generate_terrain(bbox, params)
    
    assert terrain["data"].min() >= -100
    assert terrain["data"].max() <= 4000
    assert terrain["data"].shape == (256, 256)

def test_landcover_generation(generator):
    """Test land cover generation with default parameters."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    landcover = generator.generate_landcover(bbox)
    
    assert isinstance(landcover, dict)
    assert "data" in landcover
    assert "transform" in landcover
    assert "crs" in landcover
    assert landcover["data"].shape == (256, 256)
    assert landcover["data"].dtype == np.int32

def test_landcover_generation_with_params(generator):
    """Test land cover generation with custom parameters."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    params = LandCoverParams(
        classes=["water", "urban", "forest"],
        class_weights=[0.3, 0.3, 0.4],
        patch_size_range=(10, 50),
        smoothing=1.0,
        random_state=42
    )
    
    landcover = generator.generate_landcover(bbox, params)
    
    assert landcover["data"].shape == (256, 256)
    assert set(np.unique(landcover["data"])) <= {0, 1, 2}

def test_multispectral_generation(generator):
    """Test multispectral imagery generation."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    terrain = generator.generate_terrain(bbox)
    landcover = generator.generate_landcover(bbox)
    params = AtmosphericParams(
        cloud_cover=0.3,
        cloud_type="cumulus",
        haze=0.2,
        aerosol_depth=0.1,
        sun_elevation=45.0,
        sun_azimuth=180.0
    )
    
    imagery = generator.generate_multispectral(
        bbox,
        terrain=terrain,
        landcover=landcover,
        atmospheric_params=params
    )
    
    assert isinstance(imagery, dict)
    assert "data" in imagery
    assert "transform" in imagery
    assert "crs" in imagery
    assert imagery["data"].shape == (256, 256, 4)
    assert not np.isnan(imagery["data"]).any()

def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    with pytest.raises(ValueError):
        SyntheticDataGenerator(output_size=(0, 0))
    
    with pytest.raises(ValueError):
        SyntheticDataGenerator(resolution=-1.0)
    
    with pytest.raises(ValueError):
        TerrainParams(elevation_range=(1000, -1000))
    
    with pytest.raises(ValueError):
        LandCoverParams(patch_size_range=(100, 10))

def test_device_handling():
    """Test device handling and GPU availability check."""
    import torch
    
    if torch.cuda.is_available():
        generator = SyntheticDataGenerator(device="cuda")
        assert str(generator.device) == "cuda:0"
    else:
        generator = SyntheticDataGenerator(device="cpu")
        assert str(generator.device) == "cpu"

def test_error_handling(generator):
    """Test error handling for invalid inputs."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    
    with pytest.raises(SyntheticDataError):
        generator.generate_terrain(None)
    
    with pytest.raises(SyntheticDataError):
        generator.generate_landcover(None)
    
    with pytest.raises(SyntheticDataError):
        generator.generate_multispectral(bbox, terrain=None, landcover=None)

def test_reproducibility(generator):
    """Test reproducibility with fixed seeds."""
    bbox = (-122.4194, 37.7749, -122.4094, 37.7849)
    params1 = TerrainParams(seed=42)
    params2 = TerrainParams(seed=42)
    
    terrain1 = generator.generate_terrain(bbox, params1)
    terrain2 = generator.generate_terrain(bbox, params2)
    
    np.testing.assert_array_equal(terrain1["data"], terrain2["data"]) 