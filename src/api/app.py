from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from flasgger import Swagger
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
from datetime import datetime
import uuid
import logging
from logging.handlers import RotatingFileHandler
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import mercantile
import numpy as np
from PIL import Image
import io
from shapely.geometry import shape, mapping
from src.privacy.secure_encoding import SecureAPILayer
from dotenv import load_dotenv
from functools import wraps
from werkzeug.utils import secure_filename
from google.cloud import storage
import torch

from synthetic_data_generator import SyntheticConfig, SyntheticDataGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Initialize Swagger
swagger = Swagger(app, template_file='swagger.yaml')

# Initialize secure API layer
secure_api = SecureAPILayer(os.getenv('MASTER_KEY', 'your-secure-master-key'))

# Register some API keys (in production, load from secure storage)
secure_api.register_api_key(
    os.getenv('API_KEY_FULL'),
    {'read', 'write', 'generate'}
)
secure_api.register_api_key(
    os.getenv('API_KEY_READ'),
    {'read'}
)

def require_api_key(permission):
    """Decorator to require API key with specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key or not secure_api.validate_api_key(api_key, permission):
                return jsonify({'error': 'Invalid or missing API key'}), 401
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Load configuration
config = SyntheticConfig.from_yaml('config.yaml')
generator = SyntheticDataGenerator(config)

# Cache for tile results
from cachetools import TTLCache
tile_cache = TTLCache(
    maxsize=int(os.getenv('TILE_CACHE_SIZE', 1000)),
    ttl=int(os.getenv('TILE_CACHE_TTL', 3600))
)

def get_tile_key(z: int, x: int, y: int, params: Dict[str, Any]) -> str:
    """Generate unique key for tile caching"""
    param_str = json.dumps(params, sort_keys=True)
    return f"{z}_{x}_{y}_{param_str}"

@app.route('/health')
@limiter.exempt
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/v1/generate', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def generate_synthetic():
    """Generate synthetic image with geo-privacy protection"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        prompt = request.form.get('prompt', 'A satellite view of terrain')
        negative_prompt = request.form.get('negative_prompt', '')
        protection_level = request.form.get('protection_level', 'high')
        layout_type = request.form.get('layout_type', 'grid')
        fractal_type = request.form.get('fractal_type')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp:
            file.save(temp.name)
            
        # Process image
        result = generator.process_large_image(
            input_path=temp.name,
            output_path=None,  # Don't save to disk
            prompt=prompt,
            negative_prompt=negative_prompt,
            return_array=True  # Return numpy array instead of saving
        )
        
        # Get geometry from input image
        with rasterio.open(temp.name) as src:
            bounds = src.bounds
            geometry = shape({
                'type': 'Polygon',
                'coordinates': [[
                    [bounds.left, bounds.bottom],
                    [bounds.right, bounds.bottom],
                    [bounds.right, bounds.top],
                    [bounds.left, bounds.top],
                    [bounds.left, bounds.bottom]
                ]]
            })
        
        # Cleanup
        os.unlink(temp.name)
        
        if result is None:
            return jsonify({'error': 'Failed to generate image'}), 500
            
        # Encode with geo-privacy protection
        encrypted_data, secure_metadata = secure_api.encode_tile_with_geo_privacy(
            result['image'],
            geometry,
            {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'timestamp': datetime.utcnow().isoformat()
            },
            protection_level,
            layout_type,
            fractal_type
        )
        
        # Save encrypted result
        output_path = Path('outputs') / f"{uuid.uuid4()}.enc"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
            
        return jsonify({
            'status': 'success',
            'output_path': str(output_path),
            'metadata': secure_metadata
        })
        
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/tiles/<int:z>/<int:x>/<int:y>.png')
@require_api_key('read')
def get_tile(z: int, x: int, y: int):
    """Serve encrypted XYZ tiles with geo-privacy protection"""
    try:
        # Get parameters
        image_url = request.args.get('image_url')
        if not image_url:
            return jsonify({'error': 'image_url parameter is required'}), 400
            
        access_token = request.headers.get('X-Access-Token')
        if not access_token:
            return jsonify({'error': 'Access token required'}), 401
            
        # Check cache
        cache_key = get_tile_key(z, x, y, request.args)
        if cache_key in tile_cache:
            encrypted_data, secure_metadata = tile_cache[cache_key]
            
            # Decode tile with geo-privacy
            tile_data, geometry = secure_api.decode_tile_with_geo_privacy(
                encrypted_data,
                secure_metadata,
                access_token
            )
            
            if tile_data is None:
                return jsonify({'error': 'Unauthorized'}), 401
                
            # Convert to PNG
            img_byte_arr = io.BytesIO()
            Image.fromarray(tile_data).save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return send_file(
                img_byte_arr,
                mimetype='image/png'
            )
            
        # Calculate tile bounds
        tile = mercantile.Tile(x, y, z)
        bounds = mercantile.bounds(tile)
        
        # Process tile
        with rasterio.open(image_url) as src:
            # Transform bounds to image CRS
            bounds_transformed = transform_bounds(
                CRS.from_epsg(4326),
                src.crs,
                *bounds
            )
            
            # Create geometry from bounds
            geometry = shape({
                'type': 'Polygon',
                'coordinates': [[
                    [bounds.west, bounds.south],
                    [bounds.east, bounds.south],
                    [bounds.east, bounds.north],
                    [bounds.west, bounds.north],
                    [bounds.west, bounds.south]
                ]]
            })
            
            # Generate tile
            result = generator.generate_synthetic_image(
                input_image=src.read(),
                prompt=request.args.get('prompt', 'A satellite view of terrain'),
                negative_prompt=request.args.get('negative_prompt', '')
            )
            
            if result is None:
                return jsonify({'error': 'Failed to generate tile'}), 500
                
            # Encode tile with geo-privacy
            encrypted_data, secure_metadata = secure_api.encode_tile_with_geo_privacy(
                result['image'],
                geometry,
                {
                    'z': z,
                    'x': x,
                    'y': y,
                    'bounds': bounds._asdict(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Cache result
            tile_cache[cache_key] = (encrypted_data, secure_metadata)
            
            # Decode for response
            tile_data, _ = secure_api.decode_tile_with_geo_privacy(
                encrypted_data,
                secure_metadata,
                access_token
            )
            
            if tile_data is None:
                return jsonify({'error': 'Unauthorized'}), 401
                
            # Convert to PNG
            img_byte_arr = io.BytesIO()
            Image.fromarray(tile_data).save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return send_file(
                img_byte_arr,
                mimetype='image/png'
            )
            
    except Exception as e:
        logger.error(f"Error serving tile: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/map')
def map_viewer():
    """Serve map viewer HTML"""
    return send_from_directory('templates', 'map.html')

@app.route('/api/v1/capabilities')
def get_capabilities():
    """Get server capabilities and configuration"""
    return jsonify({
        'version': '1.0.0',
        'supported_formats': ['png', 'tiff'],
        'max_image_size': 10000 * 10000,
        'tile_size': config.tile_size,
        'models': {
            'stable_diffusion': config.stable_diffusion_model,
            'controlnet': config.controlnet_model,
            'segmentation': config.segmentation_model
        },
        'endpoints': {
            'generate': '/api/v1/generate',
            'tiles': '/api/v1/tiles/{z}/{x}/{y}.png',
            'map': '/api/v1/map'
        }
    })

@app.route('/api/v1/download/<path:filename>')
@require_api_key('read')
def download_result(filename):
    """Download encrypted results with geo-privacy protection"""
    try:
        access_token = request.headers.get('X-Access-Token')
        if not access_token:
            return jsonify({'error': 'Access token required'}), 401
            
        file_path = Path('outputs') / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
            
        # Read encrypted data
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
            
        # Get metadata from request
        secure_metadata = request.get_json()
        if not secure_metadata:
            return jsonify({'error': 'Metadata required'}), 400
            
        # Decode data with geo-privacy
        image_data, geometry = secure_api.decode_tile_with_geo_privacy(
            encrypted_data,
            secure_metadata,
            access_token
        )
        
        if image_data is None:
            return jsonify({'error': 'Unauthorized'}), 401
            
        # Convert to PNG
        img_byte_arr = io.BytesIO()
        Image.fromarray(image_data).save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return send_file(
            img_byte_arr,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{filename.stem}.png"
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/plant/flow', methods=['POST'])
@limiter.limit("10 per minute")      # Apply rate limiting if needed
@require_api_key('generate')         # Apply API key requirement based on permission
def plant_flow():
    """
    Handle the generation of synthetic images with geo-privacy protection.
    ---
    tags:
      - Plant Operations
    consumes:
      - multipart/form-data
    parameters:
      - name: Secret-Key
        in: query
        type: string
        required: true
        description: User Secret Key for authentication
      - name: image
        in: formData
        type: file
        required: true
        description: Image file to upload
    responses:
      200:
        description: Successfully generated synthetic image.
        schema:
          type: object
          properties:
            status:
              type: integer
              example: 200
            message:
              type: string
              example: Success
            description:
              type: string
              example: "Description extracted from image"
            text1:
              type: string
              example: "Generated text from LLaMA model"
            scene:
              type: string
              example: "Scene description for Stable Diffusion"
            image:
              type: string
              example: "https://storage.cloud.google.com/appimage/geni1_20231005_123456.png"
      400:
        description: Bad Request - Invalid input or missing parameters.
        schema:
          type: object
          properties:
            status:
              type: integer
              example: 400
            message:
              type: string
              example: "Invalid file type."
      401:
        description: Unauthorized - Invalid or missing API key.
        schema:
          type: object
          properties:
            status:
              type: integer
              example: 401
            message:
              type: string
              example: "Unauthorized"
      500:
        description: Internal Server Error - An unexpected error occurred.
        schema:
          type: object
          properties:
            status:
              type: integer
              example: 500
            message:
              type: string
              example: "Internal Server Error"
    """
    try:
        # Clear GPU cache if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Received a request to /plant/flow")

        # === Authentication ===
        secret_key = request.args.get("Secret-Key")
        userId = request.args.get("user_id", '')

        expected_secret = app.config.get("SECRET_KEY")
        if secret_key != expected_secret:
            logger.warning("Invalid secret key provided")
            return jsonify({"status": 401, "message": "Unauthorized"}), 401

        # === Image Validation ===
        if "image" not in request.files:
            logger.warning("No image part in the request")
            return jsonify({"status": 400, "message": "No image part in request"}), 400

        image = request.files["image"]
        if image.filename == "":
            logger.warning("No selected image")
            return jsonify({"status": 400, "message": "No selected image"}), 400

        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.warning("Invalid file type uploaded")
            return jsonify({"status": 400, "message": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}), 400

        # === Load Models ===
        load_stable_diffusion_model()  # Load Stable Diffusion model
        load_llama_model()             # Load LLaMA model

        # === Save Uploaded Image ===
        upload_folder = app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(image.filename)
        upload_path = os.path.join(upload_folder, filename)
        image.save(upload_path)
        logger.info(f"Image saved to {upload_path}")

        # === Image Description (BLIP Model) ===
        description = extract_image_details(upload_path)
        logger.info(f"Extracted description: {description}")

        # === Generate Response (LLaMA Model) ===
        prompt = (
            f"From the sentence: \"{description}\", extract all living objects such as plants, animals, or any living entities. "
            f"Respond ONLY json containing comma separated living object names, without any additional text or examples."
        )
        generated_text = generate_prompt_from_caption(prompt)
        logger.info(f"Generated text: {generated_text}")

        # Extract JSON portion
        json_start = generated_text.find("{")
        if json_start == -1:
            raise ValueError("No JSON found in generated text")
        json_part = generated_text[json_start:].strip()
        response_dict = json.loads(json_part)

        # Extract based on the structure of the JSON
        living_objects = []
        for key, value in response_dict.items():
            if isinstance(value, str) and key.strip():  # Extract value if it's a string
                living_objects.append(key.strip())       # Extract the key (e.g., "tomatoes")
            elif isinstance(value, list):               # Handle lists if present
                living_objects.extend([item.strip() for item in value if isinstance(item, str)])

        # Ensure unique items
        living_objects = list(set(living_objects))

        logger.info(f"Extracted living objects: {living_objects}")

        # Create a description for Stable Diffusion
        prompt1 = (
            f"Create a description for a stable diffusion prompt where \"{living_objects}\" are seen in a farm. "
            f"Respond with only a descriptive sentence suitable for generating an image, without any extra text."
        )
        scene = generate_prompt_from_caption(prompt1)
        logger.info(f"Scene description for Stable Diffusion: {scene}")

        # === Generate Image (Stable Diffusion Model) ===
        generated_image = generate_image_from_text(scene, None)

        # === Save Generated Image ===
        image_byte_array = io.BytesIO()
        generated_image.save(image_byte_array, format='PNG', optimize=True, quality=85)

        image_byte_array.seek(0)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        unique_filename = f"geni1_{timestamp}.png"
        unique_filename1 = f"up1_{timestamp}.png"

        # === Set Up Google Cloud Storage ===
        service_account_path = "/home/jaya/syn_project/peak-sorter-432307-p8-c68a06a6da43.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

        bucket_name = "appimage"
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        blob = bucket.blob(unique_filename1)
        with open(upload_path, "rb") as image_file:
            blob.upload_from_file(image_file, content_type='image/png')

        image_url1 = f"https://storage.cloud.google.com/{bucket_name}/{unique_filename1}"

        # === Upload Image to Google Cloud Storage ===
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(image_byte_array.getvalue(), content_type='image/png')

        # Generate public URL
        image_url = f"https://storage.cloud.google.com/{bucket_name}/{unique_filename}"

        logger.info(f"Image uploaded successfully to: {image_url}")

        create_log_entry(
            user_id=userId,          # Set to None if no user
            api_id='e780',           # Replace with actual api_id if available
            text_input='',           # This is a dict and will be serialized
            image_input=image_url1,  # Assuming no input image path is needed
            image_output=image_url,  # Assuming no output image data is stored
            status=200,              # Pass as integer
            message="Success"
        )

        return jsonify({
            "status": 200,
            "message": "Success",
            "description": description,
            "text1": generated_text,
            "scene": scene,
            "image": image_url
        }), 200

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({"status": 400, "message": str(e)}), 400
    except json.JSONDecodeError as e:
        logger.error(f"JSON error: {e}")
        return jsonify({"status": 400, "message": "Failed to parse JSON response"}), 400
    except Exception as e:
        logger.error(f"Error processing /plant/flow: {e}")
        return jsonify({"status": 500, "message": "Internal Server Error"}), 500

# ====================
# + Finalizing the App
# ====================

if __name__ == '__main__':
    # Create necessary directories
    Path('outputs').mkdir(exist_ok=True)
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    ) 