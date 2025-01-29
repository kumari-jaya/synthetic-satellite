import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import transform
import pyproj
from functools import partial
import math
import random
import uuid

class GeoPrivacyEncoder:
    """Implements geo-privacy encoding techniques for protecting location data"""
    
    def __init__(self, master_salt: str = None):
        """Initialize with optional master salt for deterministic transformations"""
        self.master_salt = master_salt or str(uuid.uuid4())
        self._initialize_transforms()
    
    def _initialize_transforms(self):
        """Initialize transformation functions"""
        self.transform_funcs = {
            'grid': self._layout_transform_grid,
            'spiral': self._layout_transform_spiral,
            'cluster': self._layout_transform_cluster,
            'box': self._layout_transform_box,
            'bbox': self._layout_transform_bbox
        }
        
        self.fractal_funcs = {
            'hilbert': self._fractal_transform_hilbert,
            'spiral': self._fractal_transform_spiral,
            'koch': self._fractal_transform_koch,
            'sierpinski': self._fractal_transform_sierpinski,
            'julia': self._fractal_transform_julia
        }
    
    def encode_geometry(
        self,
        geometry: Any,
        layout_type: str = 'grid',
        fractal_type: Optional[str] = None,
        spacing: float = 200,
        protection_level: str = 'high'
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Encode geometry with privacy protection
        
        Args:
            geometry: Shapely geometry object
            layout_type: Type of layout transformation
            fractal_type: Optional fractal transformation
            spacing: Spacing between features
            protection_level: Level of privacy protection
            
        Returns:
            transformed geometry and encoding metadata
        """
        # Get transform function
        transform_func = self.transform_funcs.get(layout_type)
        if not transform_func:
            raise ValueError(f"Unknown layout type: {layout_type}")
            
        # Generate deterministic seed from geometry and master salt
        seed = int(hashlib.sha256(
            f"{self.master_salt}{geometry.wkt}".encode()
        ).hexdigest(), 16)
        random.seed(seed)
        
        # Apply layout transformation
        transformed = transform_func(geometry, spacing)
        
        # Apply fractal transformation if specified
        if fractal_type and protection_level == 'high':
            fractal_func = self.fractal_funcs.get(fractal_type)
            if fractal_func:
                transformed = fractal_func(transformed)
        
        # Create metadata
        metadata = {
            'original_type': geometry.geom_type,
            'layout_type': layout_type,
            'fractal_type': fractal_type,
            'spacing': spacing,
            'protection_level': protection_level,
            'transform_seed': seed,
            'bbox': geometry.bounds
        }
        
        return transformed, metadata
    
    def decode_geometry(
        self,
        geometry: Any,
        metadata: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Decode transformed geometry using metadata
        
        Args:
            geometry: Transformed geometry
            metadata: Encoding metadata
            
        Returns:
            Original geometry if possible, None if invalid
        """
        try:
            # Restore random seed
            random.seed(metadata['transform_seed'])
            
            # Reverse fractal transform if applied
            if metadata.get('fractal_type'):
                geometry = self._inverse_fractal_transform(
                    geometry,
                    metadata['fractal_type']
                )
            
            # Reverse layout transform
            original_type = metadata['original_type']
            bbox = metadata['bbox']
            
            # Reconstruct original geometry
            if original_type == 'Point':
                return Point(
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                )
            elif original_type in ('Polygon', 'MultiPolygon'):
                # Create approximate polygon from bbox
                return Polygon([
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3]),
                    (bbox[0], bbox[1])
                ])
            
            return geometry
            
        except Exception as e:
            print(f"Error decoding geometry: {str(e)}")
            return None
    
    def _layout_transform_grid(self, geometry: Any, spacing: float) -> Any:
        """Grid layout transformation"""
        bbox = geometry.bounds
        x_offset = random.uniform(-spacing/4, spacing/4)
        y_offset = random.uniform(-spacing/4, spacing/4)
        
        def transform_func(x, y):
            # Calculate grid position
            grid_x = math.floor(x / spacing) * spacing + x_offset
            grid_y = math.floor(y / spacing) * spacing + y_offset
            return (grid_x, grid_y)
        
        return transform(transform_func, geometry)
    
    def _layout_transform_spiral(self, geometry: Any, spacing: float) -> Any:
        """Spiral layout transformation"""
        centroid = geometry.centroid
        angle = random.uniform(0, 2 * math.pi)
        radius = spacing * math.sqrt(random.random())
        
        def transform_func(x, y):
            dx = x - centroid.x
            dy = y - centroid.y
            r = math.sqrt(dx*dx + dy*dy)
            theta = math.atan2(dy, dx) + angle
            return (
                centroid.x + r * math.cos(theta),
                centroid.y + r * math.sin(theta)
            )
        
        return transform(transform_func, geometry)
    
    def _layout_transform_cluster(self, geometry: Any, spacing: float) -> Any:
        """Cluster layout transformation"""
        clusters = 4
        cluster_radius = spacing
        cluster_id = random.randint(0, clusters-1)
        
        def transform_func(x, y):
            # Calculate cluster center
            cluster_x = (cluster_id % 2) * spacing * 3
            cluster_y = (cluster_id // 2) * spacing * 3
            
            # Add random offset within cluster
            dx = random.uniform(-cluster_radius/2, cluster_radius/2)
            dy = random.uniform(-cluster_radius/2, cluster_radius/2)
            
            return (cluster_x + dx, cluster_y + dy)
        
        return transform(transform_func, geometry)
    
    def _layout_transform_box(self, geometry: Any, spacing: float) -> Any:
        """Box layout transformation"""
        bbox = geometry.bounds
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        def transform_func(x, y):
            # Scale and shift to fit in box
            x_norm = (x - bbox[0]) / width
            y_norm = (y - bbox[1]) / height
            return (x_norm * spacing, y_norm * spacing)
        
        return transform(transform_func, geometry)
    
    def _layout_transform_bbox(self, geometry: Any, spacing: float) -> Any:
        """Bounding box based transformation"""
        bbox = geometry.bounds
        padding = spacing * 0.1
        
        def transform_func(x, y):
            # Add padding and random offset
            dx = random.uniform(-padding, padding)
            dy = random.uniform(-padding, padding)
            return (x + dx, y + dy)
        
        return transform(transform_func, geometry)
    
    def _fractal_transform_hilbert(self, geometry: Any) -> Any:
        """Hilbert curve based transformation"""
        def transform_func(x, y):
            r = math.sqrt(x*x + y*y)
            theta = math.atan2(y, x)
            return (
                r * math.cos(theta + r/100),
                r * math.sin(theta + r/100)
            )
        return transform(transform_func, geometry)
    
    def _fractal_transform_spiral(self, geometry: Any) -> Any:
        """Spiral fractal transformation"""
        def transform_func(x, y):
            r = math.sqrt(x*x + y*y)
            theta = math.atan2(y, x)
            return (
                (r + 0.5) * math.cos(theta + 2),
                (r + 0.5) * math.sin(theta + 2)
            )
        return transform(transform_func, geometry)
    
    def _fractal_transform_koch(self, geometry: Any) -> Any:
        """Koch snowflake based transformation"""
        def transform_func(x, y):
            return (
                x + 0.5 * math.sin(y/50),
                y + 0.5 * math.sin(x/50)
            )
        return transform(transform_func, geometry)
    
    def _fractal_transform_sierpinski(self, geometry: Any) -> Any:
        """Sierpinski triangle based transformation"""
        def transform_func(x, y):
            scale = 0.5
            offset = 0.3
            return (
                scale * x,
                y + offset * x if x + y > 0 else y - offset * x
            )
        return transform(transform_func, geometry)
    
    def _fractal_transform_julia(self, geometry: Any) -> Any:
        """Julia set based transformation"""
        def transform_func(x, y):
            z = complex(x/1000, y/1000)
            z = z*z + complex(0.355, 0.355)
            return (z.real * 1000, z.imag * 1000)
        return transform(transform_func, geometry)
    
    def _inverse_fractal_transform(
        self,
        geometry: Any,
        fractal_type: str
    ) -> Any:
        """Approximate inverse fractal transformation"""
        # For most fractals, exact inverse is complex
        # We approximate by scaling back to original bounds
        bbox = geometry.bounds
        
        def transform_func(x, y):
            # Normalize coordinates
            x_norm = (x - bbox[0]) / (bbox[2] - bbox[0])
            y_norm = (y - bbox[1]) / (bbox[3] - bbox[1])
            return (x_norm, y_norm)
        
        return transform(transform_func, geometry) 