import torch
from diffusers import StableDiffusionPipeline
from PIL import ImageDraw, ImageFont, Image, ImageFilter
import random
# Add any required functions for authentication, pipeline loading, and image processing
def authenticate_huggingface(token):
    if token:
        try:
            login(token)
            HfFolder.save_token(token)
            logger.info("Successfully authenticated with Hugging Face.")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise ValueError("Failed to authenticate with Hugging Face.")
    else:
        logger.error("Hugging Face API token is missing.")
        raise ValueError("Hugging Face API token is missing.")

def load_pipeline(model_id, token):
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=token
        )
        pipe.to("cuda")

        # Optionally disable safety checker
        def dummy_safety_checker(images, clip_input):
            return images, [False] * len(images)

        pipe.safety_checker = dummy_safety_checker

        logger.info(f"Successfully loaded the pipeline for model: {model_id}")
        return pipe
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise e

def remove_background(image):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        output_bytes = remove(img_bytes)

        output_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        logger.info("Background removal complete.")
        return output_image
    except Exception as e:
        logger.error(f"Error during background removal: {e}")
        raise e


def overlay_image(base_image, overlay_image, position):
    try:
        base = base_image.convert("RGBA")
        base.paste(overlay_image, position, overlay_image)
        logger.info("Overlay complete.")
        return base
    except Exception as e:
        logger.error(f"Error during image overlay: {e}")
        raise e

def synobj(pipe, prompt, negative_prompt, strength, guidance_scale, steps, init_image):
    try:
        with torch.autocast("cuda"):
            generated = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps
            ).images[0]
        logger.info("Image generation complete.")
        return generated
    except Exception as e:
        logger.error(f"Error during image generation: {e}")
        raise e



def synimg(prompt, seed):
    # Load the Stable Diffusion model (stabilityai/stable-diffusion-3.5-large)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    # Set the seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
    else:
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(f"No seed provided. Using random seed: {seed}")
    
    # Generate the image with the fixed seed
    result = pipe(prompt, generator=generator)
    image = result.images[0]

    # Convert to PIL Image if it's not already (since diffusers return PIL image)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Add watermark to the bottom-left corner (logo and text with glow)
    image_with_watermark = add_watermark(image, "Vortx", "media/logo.png")

    # Return the generated image with watermark
    return image_with_watermark

def synenv(prompt: str,image, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",num_inference_steps: int = 50,guidance_scale: float = 7.5):
    init_image = image.convert("RGB").resize((512, 512), resample=Image.LANCZOS)
        

    # Load SDXL Base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use FP16 for efficiency on GPUs
        use_safetensors=True        # More secure/faster model weights format
    )

    # Move pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    # Optional: enable memory-efficient attention (requires xformers installed)
    # pipe.enable_xformers_memory_efficient_attention()

    # Generate image
    result = pipe(prompt=prompt,
                  image=init_image,
                  num_inference_steps=50,
                  guidance_scale=7.5)
    
    return result.images[0]

def add_watermark(image, text, logo_path):
    # Convert the image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load the logo image
    logo = Image.open(logo_path)

    # Resize the logo (optional, adjust size as necessary)
    logo_width, logo_height = 50, 50  # Adjust based on your requirement
    logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing

    # Define font and size (adjust font size based on your preference)
    font = ImageFont.load_default()  # You can specify a custom font if needed

    # Calculate text size using textbbox()
    bbox = draw.textbbox((0, 0), text, font=font)
    textwidth = bbox[2] - bbox[0]
    textheight = bbox[3] - bbox[1]
    
    # Calculate total width and height (logo + text)
    total_width = logo_width + textwidth + 10  # 10px margin between logo and text
    total_height = max(logo_height, textheight)

    # Set the position for the watermark (bottom-left corner)
    width, height = image.size
    margin = 10  # Margin from the edge
    x = margin  # Left side margin
    y = height - total_height - margin  # Bottom margin

    # Create a layer for the glow effect
    glow_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Draw a slightly larger version of the logo to create the glow effect
    glow_draw = ImageDraw.Draw(glow_layer)

    # Create a glowing shadow around the logo
    logo_glow = logo.resize((logo_width + 10, logo_height + 10), Image.Resampling.LANCZOS)
    glow_layer.paste(logo_glow, (x - 5, y - 5), logo_glow)

    # Add the text glow
    glow_draw.text((x + logo_width + 15, y + (logo_height - textheight) // 2), text, font=font, fill=(255, 255, 255, 128))  # Draw a transparent white text

    # Blur the glow layer to create the glow effect
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=5))

    # Paste the glow layer onto the original image
    image = Image.alpha_composite(image.convert('RGBA'), glow_layer)

    # Paste the original logo and text on top of the glow
    image.paste(logo, (x, y), logo)  # Paste the logo
    draw = ImageDraw.Draw(image)
    draw.text((x + logo_width + 10, y + (logo_height - textheight) // 2), text, font=font, fill=(255, 255, 255))  # Add the white text

    return image.convert('RGB')  # Return the final image in RGB mode
