"""
Pytest configuration file for TileFormer tests.
"""

import os
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

@pytest.fixture(scope="session")
def test_dir():
    """Get the test directory path."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def data_dir(test_dir):
    """Get the test data directory path."""
    return test_dir / "data"

@pytest.fixture(scope="session")
def output_dir(test_dir):
    """Get the test output directory path."""
    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

@pytest.fixture(scope="session")
def test_image():
    """Create a test image with known properties."""
    # Create a simple test pattern
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add some geometric shapes
    img[64:192, 64:192] = [255, 0, 0]  # Red square
    img[96:160, 96:160] = [0, 255, 0]  # Green square
    img[112:144, 112:144] = [0, 0, 255]  # Blue square
    
    return Image.fromarray(img)

@pytest.fixture(scope="session")
def test_metadata():
    """Create test metadata."""
    return {
        "user_id": "test-user",
        "timestamp": "2024-03-20T12:00:00Z",
        "protection_level": "medium",
        "tile": {
            "z": 12,
            "x": 2048,
            "y": 2048
        },
        "generation": {
            "model": "test-model",
            "version": "1.0.0",
            "prompt": "Test prompt"
        }
    }

@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    return {
        "api": {
            "host": "localhost",
            "port": 5000,
            "debug": True
        },
        "security": {
            "master_key": "test-key",
            "token_expiry": 3600,
            "max_attempts": 3
        },
        "privacy": {
            "default_protection": "medium",
            "salt_length": 16,
            "min_spacing": 100
        },
        "generation": {
            "model_path": "models/test",
            "batch_size": 4,
            "device": "cpu"
        },
        "cache": {
            "size": 1000,
            "ttl": 3600
        }
    }

def pytest_configure(config):
    """Configure pytest for the test suite."""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and conditions."""
    # Skip slow tests unless --runslow is specified
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip GPU tests if no GPU available
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        skip_gpu = pytest.mark.skip(reason="no GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run GPU tests"
    ) 