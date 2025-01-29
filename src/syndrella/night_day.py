import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

def main():
    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    print("Loading the model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    print("Model loaded successfully.")

    # Load the local image
    input_image_path = "night_scene.jpg"  # Replace with the path to your local .jpg file
    print(f"Loading the input image from {input_image_path}...")
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    print("Image loaded and preprocessed.")

    # Define prompt
    prompt = "A bright sunny day with clear blue skies, warm sunlight, and vibrant colors.illuminate other objects like trees and houses. give them colour as well."

    # Generate the daytime scene
    print("Generating the daytime scene...")
    images = pipe(prompt=prompt, image=init_image, strength=0.7, guidance_scale=8.0).images

    # Save the result
    output_path = "daytime_scene.png"
    images[0].save(output_path)
    print(f"Daytime scene saved as: {output_path}")

if __name__ == "__main__":
    main()
