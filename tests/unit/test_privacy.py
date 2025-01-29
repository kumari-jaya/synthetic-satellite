"""
Unit tests for the privacy module.
"""

import pytest
import numpy as np
from shapely.geometry import Point, Polygon, box
from tileformer.privacy import GeoPrivacyEncoder

@pytest.fixture
def privacy_engine():
    """Create a privacy engine instance for testing."""
    return GeoPrivacyEncoder(master_salt="test-salt")

@pytest.fixture
def sample_point():
    """Create a sample point geometry."""
    return Point(0, 0)

@pytest.fixture
def sample_polygon():
    """Create a sample polygon geometry."""
    return box(-1, -1, 1, 1)

def test_grid_transform(privacy_engine, sample_point):
    """Test grid-based transformation."""
    # Apply transformation
    transformed, metadata = privacy_engine.encode_geometry(
        sample_point,
        layout_type="grid",
        spacing=100,
        protection_level="low"
    )
    
    # Check metadata
    assert metadata["original_type"] == "Point"
    assert metadata["layout_type"] == "grid"
    assert metadata["protection_level"] == "low"
    
    # Check transformation
    assert transformed.x != sample_point.x
    assert transformed.y != sample_point.y

def test_spiral_transform(privacy_engine, sample_point):
    """Test spiral transformation."""
    # Apply transformation
    transformed, metadata = privacy_engine.encode_geometry(
        sample_point,
        layout_type="spiral",
        spacing=100,
        protection_level="medium"
    )
    
    # Check metadata
    assert metadata["layout_type"] == "spiral"
    assert metadata["protection_level"] == "medium"
    
    # Check transformation
    assert transformed.x != sample_point.x
    assert transformed.y != sample_point.y

def test_fractal_transform(privacy_engine, sample_polygon):
    """Test fractal transformation."""
    # Apply transformation
    transformed, metadata = privacy_engine.encode_geometry(
        sample_polygon,
        layout_type="grid",
        fractal_type="sierpinski",
        protection_level="high"
    )
    
    # Check metadata
    assert metadata["fractal_type"] == "sierpinski"
    assert metadata["protection_level"] == "high"
    
    # Check transformation
    assert transformed.bounds != sample_polygon.bounds

def test_reversible_transform(privacy_engine, sample_point):
    """Test that transformations are reversible."""
    # Apply transformation
    transformed, metadata = privacy_engine.encode_geometry(
        sample_point,
        layout_type="grid",
        protection_level="medium"
    )
    
    # Decode transformation
    decoded = privacy_engine.decode_geometry(transformed, metadata)
    
    # Check reversibility (approximately)
    assert abs(decoded.x - sample_point.x) < 1e-10
    assert abs(decoded.y - sample_point.y) < 1e-10

def test_invalid_transform(privacy_engine, sample_point):
    """Test handling of invalid transformation type."""
    with pytest.raises(ValueError):
        privacy_engine.encode_geometry(
            sample_point,
            layout_type="invalid",
            protection_level="low"
        )

def test_protection_levels(privacy_engine, sample_polygon):
    """Test different protection levels."""
    # Test low protection
    low_transform, low_meta = privacy_engine.encode_geometry(
        sample_polygon,
        protection_level="low"
    )
    
    # Test medium protection
    med_transform, med_meta = privacy_engine.encode_geometry(
        sample_polygon,
        protection_level="medium"
    )
    
    # Test high protection
    high_transform, high_meta = privacy_engine.encode_geometry(
        sample_polygon,
        protection_level="high"
    )
    
    # Check that higher protection levels result in more transformation
    low_diff = abs(low_transform.area - sample_polygon.area)
    med_diff = abs(med_transform.area - sample_polygon.area)
    high_diff = abs(high_transform.area - sample_polygon.area)
    
    assert low_diff <= med_diff <= high_diff

def test_deterministic_transform(privacy_engine, sample_point):
    """Test that transformations are deterministic with same salt."""
    # First transformation
    trans1, meta1 = privacy_engine.encode_geometry(
        sample_point,
        layout_type="grid",
        protection_level="medium"
    )
    
    # Second transformation
    trans2, meta2 = privacy_engine.encode_geometry(
        sample_point,
        layout_type="grid",
        protection_level="medium"
    )
    
    # Check that results are identical
    assert trans1.equals(trans2)
    assert meta1["transform_seed"] == meta2["transform_seed"]

def test_different_salt(sample_point):
    """Test that different salts produce different transformations."""
    # Create two engines with different salts
    engine1 = GeoPrivacyEncoder(master_salt="salt1")
    engine2 = GeoPrivacyEncoder(master_salt="salt2")
    
    # Apply transformations
    trans1, meta1 = engine1.encode_geometry(
        sample_point,
        layout_type="grid",
        protection_level="medium"
    )
    
    trans2, meta2 = engine2.encode_geometry(
        sample_point,
        layout_type="grid",
        protection_level="medium"
    )
    
    # Check that results are different
    assert not trans1.equals(trans2)
    assert meta1["transform_seed"] != meta2["transform_seed"] 