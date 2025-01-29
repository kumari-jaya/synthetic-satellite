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
from secure_encoding import SecureAPILayer
from dotenv import load_dotenv
from functools import wraps

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

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    Path('outputs').mkdir(exist_ok=True)
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    ) 