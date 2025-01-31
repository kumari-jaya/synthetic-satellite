import sys
import os
from flask import Flask, request, jsonify, send_file, send_from_directory, redirect, current_app
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from flasgger import Swagger, swag_from
import yaml

import threading
from werkzeug.utils import secure_filename
import torch
from io import BytesIO

# Import model load functions

# Add this import for type annotations
from typing import Dict, Any, List, Optional

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

# Use preloaded models directly
from vision.process_vis import extract_image_details
from pompei.process_pr import generate_prompt_from_caption
from syndrella.process_imgen import generate_image_from_text
from geo.helper import tile_coords_to_bbox, generate_tile

from vision.process_vis import load_blip_model, unload_blip_model, unload_transformer_model
from syndrella.process_imgen import load_stable_diffusion_model, unload_stable_diffusion_model
from pompei.process_pr import load_llama_model, unload_llama_model
from pathlib import Path

from privacy.secure_encoding import SecureAPILayer  # Ensure this exists in src/privacy/secure_encoding.py
from synthetic_data_generator import SyntheticConfig, SyntheticDataGenerator  # Ensure correct path
from geo.helper import generate_tile, tile_coords_to_bbox  # Ensure correct path

# Additional Imports for New Endpoints
from vortx.core.memory import EarthMemoryStore
from vortx.core.synthesis import (SynthesisPipeline,SatelliteDataSource)
from vortx.core.data_sources import (
    
    WeatherDataSource,
    ElevationDataSource,
    LandUseDataSource,
    ClimateDataSource
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import json
import tempfile
from datetime import datetime, timedelta
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
from functools import wraps
from werkzeug.utils import secure_filename
from google.cloud import storage
import torch
import base64



# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')  # Set upload folder in config

# Add a secret key for session management
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Swagger configuration
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Synthetic Satellite API",
        "description": "API for generating and managing synthetic satellite imagery",
        "version": "1.0.0",
        "contact": {
            "name": "API Support",
            "url": "http://www.yourwebsite.com",
        }
    },
    "basePath": "/api/v1",
    "schemes": [
        "http",
        "https"
    ],
    "securityDefinitions": {
        "ApiKeyAuth": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header"
        }
    },
    "security": [
        {
            "ApiKeyAuth": []
        }
    ]
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

# Initialize Swagger
swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

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

# ====================
# + Existing Routes
# ====================

@app.route('/')
def index():
    """Root endpoint - redirect to API documentation"""
    return redirect('/apidocs/')

@app.route('/health')
@limiter.exempt
def health_check():
    """
    Health Check Endpoint
    ---
    responses:
      200:
        description: Returns the health status of the application
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            timestamp:
              type: string
              format: date-time
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/v1/generate', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def generate_synthetic():
    """
    Generate Synthetic Satellite Image
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - image_parameters
          properties:
            image_parameters:
              type: object
              required:
                - resolution
                - format
              properties:
                resolution:
                  type: integer
                  example: 1024
                format:
                  type: string
                  example: png
                # Add other relevant parameters as needed
    responses:
      200:
        description: Successfully generated synthetic image
        schema:
          type: object
          properties:
            image_url:
              type: string
              example: http://34.45.181.99:5000/api/v1/download/generated_image.png
      400:
        description: Invalid input parameters
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Missing 'image_parameters' in request body"
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Invalid or missing API key"
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
        data = request.get_json()
        if not data or 'image_parameters' not in data:
            return jsonify({"error": "Missing 'image_parameters' in request body"}), 400

        # Extract parameters
        image_params = data['image_parameters']
        resolution = image_params.get('resolution', 1024)
        format_ = image_params.get('format', 'png')

        # Your implementation to generate the synthetic image
        # For example:
        # image_url = generator.generate_image(resolution, format_)
        image_url = "http://34.45.181.99:5000/api/v1/download/generated_image.png"  # Placeholder

        return jsonify({"image_url": image_url}), 200

    except Exception as e:
        logger.error(f"Error in generate_synthetic: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/tiles/<int:z>/<int:x>/<int:y>.png')
@require_api_key('read')
def get_tile(z: int, x: int, y: int):
    """
    Retrieve Encrypted XYZ Tile
    ---
    parameters:
      - name: z
        in: path
        type: integer
        required: true
        description: Zoom level
      - name: x
        in: path
        type: integer
        required: true
        description: Tile's X coordinate
      - name: y
        in: path
        type: integer
        required: true
        description: Tile's Y coordinate
    responses:
      200:
        description: Returns the encrypted tile image
        content:
          image/png:
            schema:
              type: string
              format: binary
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Invalid or missing API key"
      404:
        description: Tile not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Tile not found"
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
        # Your implementation to retrieve the tile
        # For example:
        # tile_data = generator.get_tile(z, x, y)
        # if not tile_data:
        #     return jsonify({"error": "Tile not found"}), 404
        # return send_file(tile_data, mimetype='image/png')
        return send_file(io.BytesIO(b''), mimetype='image/png')  # Placeholder

    except Exception as e:
        logger.error(f"Error in get_tile: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/map')
def map_viewer():
    """
    Serve Map Viewer HTML
    ---
    responses:
      200:
        description: Returns the map viewer HTML page
        content:
          text/html:
            schema:
              type: string
              example: "<!DOCTYPE html>..."
      404:
        description: HTML template not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Template not found"
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
        return send_from_directory('templates', 'map.html')
    except Exception as e:
        logger.error(f"Error in map_viewer: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/capabilities')
def get_capabilities():
    """
    Get Server Capabilities and Configuration
    ---
    responses:
      200:
        description: Returns the server capabilities and configuration
        schema:
          type: object
          properties:
            version:
              type: string
              example: "1.0.0"
            supported_formats:
              type: array
              items:
                type: string
              example: ["png", "tiff"]
            max_image_size:
              type: integer
              example: 100000000
            tile_size:
              type: integer
              example: 256
            models:
              type: object
              properties:
                stable_diffusion:
                  type: string
                  example: "stable_diffusion_model_v1"
                controlnet:
                  type: string
                  example: "controlnet_model_v2"
                segmentation:
                  type: string
                  example: "segmentation_model_v3"
            endpoints:
              type: object
              properties:
                generate:
                  type: string
                  example: "/api/v1/generate"
                tiles:
                  type: string
                  example: "/api/v1/tiles/{z}/{x}/{y}.png"
                map:
                  type: string
                  example: "/api/v1/map"
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
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
    except Exception as e:
        logger.error(f"Error in get_capabilities: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/download/<path:filename>')
@require_api_key('read')
def download_result(filename):
    """
    Download Encrypted Results with Geo-Privacy Protection
    ---
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Name of the file to download
    responses:
      200:
        description: Returns the encrypted file for download
        content:
          application/octet-stream:
            schema:
              type: string
              format: binary
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Invalid or missing API key"
      404:
        description: File not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: "File not found"
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
        # Your implementation to handle file download
        # For example:
        # file_path = os.path.join('outputs', filename)
        # if not os.path.exists(file_path):
        #     return jsonify({"error": "File not found"}), 404
        # return send_file(file_path, as_attachment=True)
        return send_file(io.BytesIO(b''), mimetype='application/octet-stream')  # Placeholder
    except Exception as e:
        logger.error(f"Error in download_result: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/locen', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def locen():
    """
    Handle Generation of Synthetic Images with Geo-Privacy Protection
    """
    try:
        data = request.get_json()
        if not data or 'flow_parameters' not in data:
            return jsonify({"error": "Missing 'flow_parameters' in request body"}), 400

        flow_params = data['flow_parameters']
        input_data = flow_params.get('input_data')
        if not input_data:
            return jsonify({"error": "Missing 'input_data' in 'flow_parameters'"}), 400

        # Handle image upload if present
        image = request.files.get('image')
        if image:
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)

            filename = secure_filename(image.filename)
            upload_path = os.path.join(upload_folder, filename)
            image.save(upload_path)
        else:
            upload_path = None  # Handle accordingly

        # === Load Models ===
        load_stable_diffusion_model()  # Load Stable Diffusion model
        load_llama_model()

        # === Image Description (BLIP Model) ===
        if upload_path:
            description = extract_image_details(upload_path)
            logger.info(f"Extracted description: {description}")
        else:
            description = "No image provided."
            logger.info("No image uploaded for description extraction.")

        # === Generate Response (LLaMA Model) ===
        prompt = (
            f"From the sentence: \"{description}\", extract all living objects such as plants, animals, or any living entities. "
            f"Respond ONLY with JSON containing comma-separated living object names, without any additional text or examples."
        )
        generated_text = generate_prompt_from_caption(prompt)
        logger.info(f"Generated text: {generated_text}")

        # Extract JSON portion
        json_start = generated_text.find("{")
        if json_start == -1:
            raise ValueError("No JSON found in generated text")

        # Parse the JSON
        json_part = generated_text[json_start:].strip()
        response_dict = json.loads(json_part)

        # Extract based on the structure of the JSON
        living_objects = []
        for key, value in response_dict.items():
            if isinstance(value, str) and key.strip():  # Extract value if it's a string
                living_objects.append(key.strip())  # Extract the key (e.g., "tomatoes")
            elif isinstance(value, list):  # Handle lists if present
                living_objects.extend([item.strip() for item in value if isinstance(item, str)])

        # Ensure unique items
        living_objects = list(set(living_objects))

        logger.info(f"Extracted living objects: {living_objects}")

        # Create a description for Stable Diffusion
        prompt1 = (
            f"Create a description for a stable diffusion prompt where \"{', '.join(living_objects)}\" are seen in a farm. "
            f"Respond with only a descriptive sentence suitable for generating an image, without any extra text."
        )
        scene = generate_prompt_from_caption(prompt1)

        logger.info(f"Scene description for Stable Diffusion: {scene}")

        # Handle potential extra content
        content_start = scene.find("\n\n")
        if content_start != -1:
            extracted_content = scene[content_start + 2:].strip()
        else:
            extracted_content = scene

        # === Generate Image (Stable Diffusion Model) ===
        generated_image = generate_image_from_text(extracted_content, None)

        # === Save Generated Image ===
        image_byte_array = io.BytesIO()
        generated_image.save(image_byte_array, format='PNG', optimize=True, quality=85)

        image_byte_array.seek(0)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        unique_filename = f"geni1_{timestamp}.png"
        unique_filename1 = f"up1_{timestamp}.png"

        # === Set Up Google Cloud Storage ===
        service_account_path = os.getenv("SERVICE_ACCOUNT")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

        bucket_name = os.getenv("BUCKET_NAME")
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        # Upload original image if provided
        if upload_path:
            blob = bucket.blob(unique_filename1)
            with open(upload_path, "rb") as image_file:
                blob.upload_from_file(image_file, content_type='image/png')

            image_url1 = f"https://storage.cloud.google.com/{bucket_name}/{unique_filename1}"
        else:
            image_url1 = None

        # === Upload Generated Image to Google Cloud Storage ===
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(image_byte_array.getvalue(), content_type='image/png')

        # Generate public URL
        image_url = f"https://storage.cloud.google.com/{bucket_name}/{unique_filename}"

        # Unload models after use to free memory
        unload_stable_diffusion_model()
        unload_llama_model()

        return jsonify({
            "status": 200,
            "message": "Success",
            "description": description,
            "text1": generated_text,
            "scene": scene,
            "image": image_url,
            "uploaded_image_url": image_url1
        }), 200

    except Exception as e:
        logger.error(f"Error in locen: {str(e)}")
        return jsonify({"error": str(e)}), 500  
 

@app.route("/tileformer", methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def tileformer():
    """
    Generate Tile from B04 and B08 Band Images
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - band_images
            - bounding_box
          properties:
            band_images:
              type: object
              required:
                - B04
                - B08
              properties:
                B04:
                  type: string
                  format: url
                  example: "http://example.com/image_b04.png"
                B08:
                  type: string
                  format: url
                  example: "http://example.com/image_b08.png"
            bounding_box:
              type: object
              required:
                - minx
                - miny
                - maxx
                - maxy
              properties:
                minx:
                  type: number
                  example: -180.0
                miny:
                  type: number
                  example: -90.0
                maxx:
                  type: number
                  example: 180.0
                maxy:
                  type: number
                  example: 90.0
            tile_size:
              type: integer
              example: 256
            algorithm:
              type: string
              example: "transformer"
    responses:
      200:
        description: Successfully generated tile image
        content:
          image/png:
            schema:
              type: string
              format: binary
      400:
        description: Invalid input parameters
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Missing required parameters"
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Invalid or missing API key"
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request payload must be in JSON format"}), 400

        # Validate band image URLs
        band_images = data.get("band_images")
        if not band_images or 'B04' not in band_images or 'B08' not in band_images:
            return jsonify({"error": "Both 'B04' and 'B08' image URLs are required"}), 400

        b04_url = band_images['B04']
        b08_url = band_images['B08']

        # Validate and parse bounding box parameters
        bounding_box = data.get("bounding_box", {})
        try:
            minx = float(bounding_box.get("minx", -180))
            miny = float(bounding_box.get("miny", -90))
            maxx = float(bounding_box.get("maxx", 180))
            maxy = float(bounding_box.get("maxy", 90))
        except ValueError:
            return jsonify({"error": "Bounding box parameters must be numeric"}), 400

        tile_size = int(data.get('tile_size', 256))
        algorithm = data.get('algorithm', 'transformer')

        bbox = [minx, miny, maxx, maxy]

        # Generate and serve the tile
        img_byte_arr = generate_tile(b04_url, b08_url, bbox, tile_size, algorithm)

        if img_byte_arr:
            img_byte_arr.seek(0)
            return send_file(img_byte_arr, mimetype='image/png')
        else:
            return jsonify({"error": f"Unsupported algorithm '{algorithm}'"}), 400

    except Exception as e:
        logger.error(f"Error in tileformer: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ====================
# + New Memory Analysis Endpoints
# ====================

# Initialize services for memory APIs
memory_store = EarthMemoryStore(Path("/app/data/memories"))

data_sources_memory = [
    SatelliteDataSource(
        name="sentinel2",
        resolution=10.0,
        bands=["B02", "B03", "B04", "B08"],  # RGB + NIR
        data_path=Path("/app/data/satellite")
    )
    # Add other data sources if needed
]

pipeline_memory = SynthesisPipeline(
    data_sources=data_sources_memory,
    memory_store=memory_store
)

@app.route('/api/v1/memory/query', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def query_memories():
    """
    Query memories by location and time range.
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - latitude
            - longitude
          properties:
            latitude:
              type: number
              format: float
              example: 34.05
            longitude:
              type: number
              format: float
              example: -118.25
            start_time:
              type: string
              format: date-time
              example: "2023-01-01T00:00:00Z"
            end_time:
              type: string
              format: date-time
              example: "2023-12-31T23:59:59Z"
            limit:
              type: integer
              example: 5
    responses:
      200:
        description: Successfully queried memories
        schema:
          type: array
          items:
            type: object
            properties:
              coordinates:
                type: array
                items:
                  type: number
                example: [34.05, -118.25]
              timestamp:
                type: string
                format: date-time
              metadata:
                type: object
              embedding:
                type: array
                items:
                  type: number
      400:
        description: Invalid input parameters
        schema:
          type: object
          properties:
            error:
              type: string
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        data = request.get_json()
        required_fields = ['latitude', 'longitude']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        latitude = data['latitude']
        longitude = data['longitude']
        start_time_str = data.get('start_time')
        end_time_str = data.get('end_time')
        limit = data.get('limit', 5)

        # Parse datetime strings if provided
        time_range = None
        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                time_range = (start_time, end_time)
            except ValueError:
                return jsonify({"error": "Invalid datetime format"}), 400

        # Query memories
        memories = memory_store.query_memories(
            coordinates=(latitude, longitude),
            time_range=time_range,
            k=limit
        )

        # Prepare response
        response = []
        for mem in memories:
            response.append({
                "coordinates": mem["coordinates"],
                "timestamp": mem["timestamp"].isoformat(),
                "metadata": mem["metadata"],
                "embedding": mem["embedding"].tolist()
            })

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in query_memories: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/memory/process', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def process_location():
    """
    Process a new location and store it in memory.
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - latitude
            - longitude
            - timestamp
          properties:
            latitude:
              type: number
              format: float
              example: 34.05
            longitude:
              type: number
              format: float
              example: -118.25
            timestamp:
              type: string
              format: date-time
              example: "2023-06-15T12:00:00Z"
            metadata:
              type: object
              example: {"additional_info": "Sample metadata"}
    responses:
      200:
        description: Successfully processed location
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            embedding:
              type: array
              items:
                type: number
            metadata:
              type: object
      400:
        description: Invalid input parameters
        schema:
          type: object
          properties:
            error:
              type: string
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        data = request.get_json()
        required_fields = ['latitude', 'longitude', 'timestamp']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        latitude = data['latitude']
        longitude = data['longitude']
        timestamp_str = data['timestamp']
        metadata = data.get('metadata')

        # Parse datetime string
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid datetime format"}), 400

        # Process location
        result = pipeline_memory.process_location(
            coordinates=(latitude, longitude),
            timestamp=timestamp,
            metadata=metadata
        )

        return jsonify({
            "status": "success",
            "embedding": result["embedding"].tolist(),
            "metadata": result["metadata"]
        }), 200

    except Exception as e:
        logger.error(f"Error in process_location: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/memory/process_time_series', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key('generate')
def process_time_series():
    """
    Process a location across a time range.
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - latitude
            - longitude
            - start_time
            - end_time
          properties:
            latitude:
              type: number
              format: float
              example: 34.05
            longitude:
              type: number
              format: float
              example: -118.25
            start_time:
              type: string
              format: date-time
              example: "2023-01-01T00:00:00Z"
            end_time:
              type: string
              format: date-time
              example: "2023-12-31T23:59:59Z"
            interval_days:
              type: integer
              example: 1
    responses:
      200:
        description: Successfully processed time series
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            count:
              type: integer
              example: 365
            results:
              type: array
              items:
                type: object
                properties:
                  timestamp:
                    type: string
                    format: date-time
                  embedding:
                    type: array
                    items:
                      type: number
                  metadata:
                    type: object
      400:
        description: Invalid input parameters
        schema:
          type: object
          properties:
            error:
              type: string
      401:
        description: Unauthorized - Invalid or missing API key
        schema:
          type: object
          properties:
            error:
              type: string
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        data = request.get_json()
        required_fields = ['latitude', 'longitude', 'start_time', 'end_time']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        latitude = data['latitude']
        longitude = data['longitude']
        start_time_str = data['start_time']
        end_time_str = data['end_time']
        interval_days = data.get('interval_days', 1)

        # Parse datetime strings
        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid datetime format"}), 400

        # Process time series data
        results = pipeline_memory.process_time_series(
            coordinates=(latitude, longitude),
            time_range=(start_time, end_time),
            interval_days=interval_days
        )

        # Prepare response
        response_results = []
        for result in results:
            response_results.append({
                "timestamp": result["metadata"]["timestamp"].isoformat(),
                "embedding": result["embedding"].tolist(),
                "metadata": result["metadata"]
            })

        return jsonify({
            "status": "success",
            "count": len(results),
            "results": response_results
        }), 200

    except Exception as e:
        logger.error(f"Error in process_time_series: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/memory/health', methods=['GET'])
@limiter.exempt
def memory_health_check():
    """
    Check the health of the memory service.
    ---
    responses:
      200:
        description: Returns the health status of the memory service
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            memory_count:
              type: integer
              example: 1500
      500:
        description: Internal Server Error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Detailed error message"
    """
    try:
        return jsonify({
            "status": "healthy",
            "memory_count": len(memory_store.memory_index)
        }), 200
    except Exception as e:
        logger.error(f"Error in memory_health_check: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
