"""
Unit tests for the synthetic generation module.
"""

import pytest
import numpy as np
from PIL import Image
from tileformer.generation import SyntheticGenerator
from tileformer.generation.models import GenerationModel
from tileformer.generation.prompts import PromptGenerator

@pytest.fixture
def generator():
    """Create a synthetic generator instance for testing."""
    return SyntheticGenerator()

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[64:192, 64:192] = 255  # White square in middle
    return Image.fromarray(img)

@pytest.fixture
def sample_prompt():
    """Create a sample prompt for testing."""
    return {
        "style": "satellite",
        "features": ["buildings", "roads", "vegetation"],
        "time_of_day": "day",
        "season": "summer"
    }

def test_basic_generation(generator):
    """Test basic image generation."""
    # Generate synthetic image
    result = generator.generate(
        prompt="A satellite view of an urban area with buildings and roads",
        size=(256, 256)
    )
    
    # Check result
    assert isinstance(result, Image.Image)
    assert result.size == (256, 256)
    assert result.mode == "RGB"

def test_conditional_generation(generator, sample_image):
    """Test conditional image generation."""
    # Generate with condition
    result = generator.generate_conditional(
        condition_image=sample_image,
        prompt="Transform this layout into a satellite view",
        strength=0.7
    )
    
    # Check result
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size
    assert result.mode == "RGB"

def test_batch_generation(generator):
    """Test batch generation of images."""
    # Generate batch
    results = generator.generate_batch(
        prompts=[
            "A satellite view of an urban area",
            "A satellite view of a rural area"
        ],
        size=(256, 256),
        batch_size=2
    )
    
    # Check results
    assert len(results) == 2
    for img in results:
        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)

def test_style_transfer(generator, sample_image):
    """Test style transfer generation."""
    # Apply style transfer
    result = generator.apply_style(
        source_image=sample_image,
        style="satellite",
        strength=0.8
    )
    
    # Check result
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size

def test_prompt_generation():
    """Test prompt generation."""
    prompt_gen = PromptGenerator()
    
    # Generate prompt
    prompt = prompt_gen.generate(
        style="satellite",
        features=["buildings", "roads"],
        time="day",
        season="summer"
    )
    
    # Check prompt
    assert isinstance(prompt, str)
    assert "satellite" in prompt.lower()
    assert "buildings" in prompt.lower()
    assert "roads" in prompt.lower()

def test_model_loading():
    """Test model loading and initialization."""
    model = GenerationModel()
    
    # Check model initialization
    assert model.is_initialized()
    assert model.get_device() in ["cuda", "cpu"]

def test_invalid_generation(generator):
    """Test handling of invalid generation parameters."""
    # Invalid size
    with pytest.raises(ValueError):
        generator.generate(
            prompt="A satellite view",
            size=(0, 0)
        )
    
    # Invalid prompt
    with pytest.raises(ValueError):
        generator.generate(
            prompt="",
            size=(256, 256)
        )

def test_generation_with_metadata(generator):
    """Test generation with metadata."""
    # Generate with metadata
    result, metadata = generator.generate_with_metadata(
        prompt="A satellite view of an urban area",
        size=(256, 256),
        include_metadata=True
    )
    
    # Check result and metadata
    assert isinstance(result, Image.Image)
    assert isinstance(metadata, dict)
    assert "generation_params" in metadata
    assert "timestamp" in metadata
    assert "model_version" in metadata

def test_generation_parameters(generator):
    """Test different generation parameters."""
    # Test different parameters
    results = []
    for temp in [0.7, 1.0]:
        for steps in [20, 30]:
            result = generator.generate(
                prompt="A satellite view",
                size=(256, 256),
                temperature=temp,
                num_inference_steps=steps
            )
            results.append(result)
    
    # Check results
    assert len(results) == 4
    for img in results:
        assert isinstance(img, Image.Image)

def test_generation_interruption(generator):
    """Test generation interruption handling."""
    # Set up interruption
    def interrupt_callback():
        return True
    
    # Attempt generation with interruption
    result = generator.generate(
        prompt="A satellite view",
        size=(256, 256),
        callback=interrupt_callback
    )
    
    # Check that generation was interrupted gracefully
    assert result is None 