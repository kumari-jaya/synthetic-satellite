import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

def main():
    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    print("Loading the model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    print("Model loaded successfully.")

    # Load the local daytime image
    input_image_path = "day_scene1.jpg"  # Replace with your local input image path
    print(f"Loading the input image from {input_image_path}...")
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    print("Image loaded and resized.")

    # Refined prompt: No green glow, focus on darkness
    prompt = (
        "Existing image from day to night scene"
    )

    # Generate the nighttime image with higher strength
    print("Generating the strict dark nighttime scene...")
    images = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.98,  # Max overwrite
        guidance_scale=15.0  # Even stricter adherence to the prompt
    ).images

    # Save the result
    output_path = "final_dark_nighttime_scene.png"
    images[0].save(output_path)
    print(f"Nighttime scene saved as: {output_path}")

if __name__ == "__main__":
    main()
