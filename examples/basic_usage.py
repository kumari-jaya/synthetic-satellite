"""
Basic example showing how to use TileFormer for privacy-preserving synthetic map generation.
"""

import os
from pathlib import Path
import rasterio
import numpy as np
from PIL import Image

from tileformer.privacy import GeoPrivacyEncoder
from tileformer.core.security import SecureImageEncoder
from tileformer.generation import SyntheticGenerator

def main():
    # Initialize components
    privacy_engine = GeoPrivacyEncoder(master_salt="example-salt")
    security = SecureImageEncoder(master_key="example-key")
    generator = SyntheticGenerator()
    
    # Load example image
    input_path = Path("examples/data/sample.tif")
    if not input_path.exists():
        print(f"Please place a sample GeoTIFF file at {input_path}")
        return
        
    # Read input image and geometry
    with rasterio.open(input_path) as src:
        image = src.read()
        bounds = src.bounds
        
        # Create geometry from bounds
        from shapely.geometry import box
        geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        
        # Generate synthetic image
        synthetic = generator.generate(
            image,
            prompt="A satellite view of terrain",
            negative_prompt="clouds, artifacts"
        )
        
        # Apply privacy protection
        protected_geom, geo_metadata = privacy_engine.encode_geometry(
            geometry,
            layout_type="grid",
            fractal_type="sierpinski",
            protection_level="high"
        )
        
        # Encrypt result
        encrypted_data, secure_metadata = security.encode_with_geo_privacy(
            synthetic,
            protected_geom,
            {
                'bounds': bounds._asdict(),
                'crs': src.crs.to_string(),
                'timestamp': '2024-01-01T00:00:00Z'
            },
            protection_level="high",
            layout_type="grid",
            fractal_type="sierpinski"
        )
        
        # Save encrypted result
        output_path = Path("examples/output")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "encrypted.bin", "wb") as f:
            f.write(encrypted_data)
            
        # Save metadata
        import json
        with open(output_path / "metadata.json", "w") as f:
            json.dump(secure_metadata, f, indent=2)
            
        print("Successfully generated and encrypted synthetic data!")
        print(f"Results saved in {output_path}")
        
        # Example of decoding (with proper access token)
        access_token = secure_metadata['access_token']
        decoded_image, decoded_geom = security.decode_with_geo_privacy(
            encrypted_data,
            secure_metadata,
            access_token
        )
        
        if decoded_image is not None:
            # Save decoded image
            Image.fromarray(decoded_image).save(output_path / "decoded.png")
            print("Successfully decoded the result!")
        else:
            print("Failed to decode the result.")

if __name__ == "__main__":
    main() 