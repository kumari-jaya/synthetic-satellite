import os
import logging
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import random

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global reference for the preloaded pipeline
pipe = None

def load_stable_diffusion_model():
    """
    Preloads the Stable Diffusion model into the global `pipe` variable.
    """
    global pipe

    if pipe is not None:
        logger.info("Stable Diffusion model already loaded; skipping.")
        return

    hf_cache_dir = os.getenv("CACHE_DIR", "/home/jaya/.cache/huggingface")
    stable_diffusion_model = os.getenv("STABLE_DIFFUSION_MODEL", "CompVis/stable-diffusion-v1-4")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Stable Diffusion model '{stable_diffusion_model}' on device: {device}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            stable_diffusion_model,
            variant="fp16",  # Updated from revision="fp16"
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir=hf_cache_dir,
        )
        pipe = pipe.to(device)
        logger.info("Stable Diffusion model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Stable Diffusion model: {e}")
        pipe = None
        raise RuntimeError("Failed to load Stable Diffusion model. Ensure proper environment setup and access.") from e

def unload_stable_diffusion_model():
    """
    Unloads the Stable Diffusion model from memory and clears the GPU cache.
    """
    global pipe
    if pipe:
        del pipe
        pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Stable Diffusion model unloaded and GPU cache cleared.")

def generate_image_from_text(prompt: str, seed: int = None) -> Image.Image:
    """
    Generates an image from the given text prompt using the preloaded Stable Diffusion pipeline.

    Args:
        prompt (str): The text prompt to generate the image.
        seed (int, optional): Seed for reproducibility. If None, a random seed is used.

    Returns:
        PIL.Image.Image: The generated image.
    """
    if pipe is None:
        raise RuntimeError("Stable Diffusion pipeline is not loaded. Call load_stable_diffusion_model() first.")

    try:
        #load_stable_diffusion_model()
        if not prompt:
            raise ValueError("Prompt cannot be empty for image generation.")

        if seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            logger.info(f"Using provided seed: {seed}")
        else:
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            logger.info(f"No seed provided. Using random seed: {seed}")

        logger.info(f"Generating image with prompt: '{prompt}'")
        image = pipe(prompt, generator=generator).images[0]
        image1 = add_watermark(image,'/home/jaya/syn_project/vortxai_logo.jpeg')
        return image1

    except Exception as e:
        logger.error(f"Error during image generation: {e}")
        raise RuntimeError("Failed to generate image from prompt.") from e

def add_watermark(image: Image.Image, logo_path: str) -> Image.Image:
    """
    Adds a watermark with a logo to the given image.

    Args:
        image (PIL.Image.Image): The original image.
        logo_path (str): Path to the logo image.

    Returns:
        PIL.Image.Image: The image with the watermark.
    """
    try:
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Create a new transparent layer for the watermark
        watermark_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))

        # Add the logo to the bottom-left corner
        if os.path.exists(logo_path):
            logo = Image.open(logo_path).convert("RGBA")
            logo_size = (50, 50)  # Adjust size as needed
            logo.thumbnail(logo_size)  # Removed `Image.ANTIALIAS`

            margin = 10  # Margin from edges
            logo_position = (margin, image.height - logo.height - margin)
            watermark_layer.paste(logo, logo_position, logo)

        # Combine the original image with the watermark layer
        watermarked_image = Image.alpha_composite(image, watermark_layer)
        return watermarked_image.convert("RGB")

    except Exception as e:
        logger.error(f"Error adding watermark: {e}")
        raise RuntimeError("Failed to add watermark to the image.") from e
