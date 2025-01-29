"""
Unit tests for the secure encoding module.
"""

import pytest
import numpy as np
from PIL import Image
from shapely.geometry import Point
from tileformer.core.secure_encoding import SecureImageEncoder, SecureAPILayer

@pytest.fixture
def secure_encoder():
    """Create a secure encoder instance for testing."""
    return SecureImageEncoder(master_key="test-key")

@pytest.fixture
def secure_api():
    """Create a secure API layer instance for testing."""
    return SecureAPILayer(master_key="test-key")

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # White square in middle
    return Image.fromarray(img)

@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "timestamp": "2024-03-20T12:00:00Z",
        "user_id": "test-user",
        "protection_level": "medium"
    }

def test_image_encryption(secure_encoder, sample_image, sample_metadata):
    """Test basic image encryption and decryption."""
    # Encrypt image
    encrypted_data, enc_metadata = secure_encoder.encode_image(
        sample_image,
        sample_metadata
    )
    
    # Check encrypted data
    assert isinstance(encrypted_data, bytes)
    assert len(encrypted_data) > 0
    
    # Check metadata
    assert "encryption_iv" in enc_metadata
    assert "encryption_salt" in enc_metadata
    assert enc_metadata["user_id"] == sample_metadata["user_id"]
    
    # Decrypt image
    decrypted_image = secure_encoder.decode_image(encrypted_data, enc_metadata)
    
    # Check decrypted image
    assert isinstance(decrypted_image, Image.Image)
    assert decrypted_image.size == sample_image.size

def test_geo_privacy_encoding(secure_encoder, sample_image):
    """Test encoding with geo-privacy protection."""
    geometry = Point(0, 0)
    
    # Encode with geo-privacy
    encrypted_data, metadata = secure_encoder.encode_with_geo_privacy(
        sample_image,
        geometry,
        {"user_id": "test-user"},
        protection_level="medium",
        layout_type="grid"
    )
    
    # Check metadata
    assert "geo_privacy" in metadata
    assert metadata["geo_privacy"]["layout_type"] == "grid"
    assert metadata["geo_privacy"]["protection_level"] == "medium"
    
    # Decode with geo-privacy
    decrypted_image, decoded_geometry = secure_encoder.decode_with_geo_privacy(
        encrypted_data,
        metadata
    )
    
    # Check results
    assert isinstance(decrypted_image, Image.Image)
    assert isinstance(decoded_geometry, Point)

def test_unauthorized_access(secure_encoder, sample_image, sample_metadata):
    """Test unauthorized access attempts."""
    # Encrypt with one key
    encrypted_data, metadata = secure_encoder.encode_image(
        sample_image,
        sample_metadata
    )
    
    # Try to decrypt with different key
    wrong_encoder = SecureImageEncoder(master_key="wrong-key")
    decrypted = wrong_encoder.decode_image(encrypted_data, metadata)
    
    # Should fail or return None
    assert decrypted is None

def test_api_layer_tile_encoding(secure_api, sample_image):
    """Test tile encoding through API layer."""
    # Encode tile
    encrypted_tile, metadata = secure_api.encode_tile(
        sample_image,
        z=12, x=2048, y=2048,
        user_id="test-user"
    )
    
    # Check metadata
    assert "tile_z" in metadata
    assert metadata["tile_z"] == 12
    assert metadata["tile_x"] == 2048
    assert metadata["tile_y"] == 2048
    
    # Decode tile
    decrypted_tile = secure_api.decode_tile(
        encrypted_tile,
        metadata,
        user_id="test-user"
    )
    
    # Check result
    assert isinstance(decrypted_tile, Image.Image)
    assert decrypted_tile.size == sample_image.size

def test_tile_cache(secure_api, sample_image):
    """Test tile caching functionality."""
    # First request
    tile1, meta1 = secure_api.encode_tile(
        sample_image,
        z=12, x=2048, y=2048,
        user_id="test-user"
    )
    
    # Second request (should hit cache)
    tile2, meta2 = secure_api.encode_tile(
        sample_image,
        z=12, x=2048, y=2048,
        user_id="test-user"
    )
    
    # Check that cached result is used
    assert tile1 == tile2
    assert meta1["cache_hit"] is False
    assert meta2["cache_hit"] is True

def test_metadata_validation(secure_api, sample_image):
    """Test metadata validation."""
    # Missing required fields
    with pytest.raises(ValueError):
        secure_api.encode_tile(
            sample_image,
            z=12, x=2048, y=2048,
            user_id=None  # Required field
        )
    
    # Invalid zoom level
    with pytest.raises(ValueError):
        secure_api.encode_tile(
            sample_image,
            z=-1, x=0, y=0,  # Invalid zoom
            user_id="test-user"
        )

def test_tile_geo_privacy(secure_api, sample_image):
    """Test tile encoding with geo-privacy."""
    # Encode tile with geo-privacy
    encrypted_tile, metadata = secure_api.encode_tile_with_geo_privacy(
        sample_image,
        z=12, x=2048, y=2048,
        user_id="test-user",
        protection_level="high",
        layout_type="spiral"
    )
    
    # Check metadata
    assert "geo_privacy" in metadata
    assert metadata["geo_privacy"]["protection_level"] == "high"
    assert metadata["geo_privacy"]["layout_type"] == "spiral"
    
    # Decode tile
    decrypted_tile = secure_api.decode_tile_with_geo_privacy(
        encrypted_tile,
        metadata,
        user_id="test-user"
    )
    
    # Check result
    assert isinstance(decrypted_tile, Image.Image)
    assert decrypted_tile.size == sample_image.size 