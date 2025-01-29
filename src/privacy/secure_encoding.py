import hmac
import hashlib
import base64
import json
from typing import Set, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import io
from shapely.geometry import shape, mapping

class SecureAPILayer:
    def __init__(self, master_key: str):
        self.master_key = master_key
        self.api_keys: Dict[str, Set[str]] = {}

    def register_api_key(self, api_key: str, permissions: Set[str]) -> None:
        """Register an API key with specific permissions"""
        self.api_keys[api_key] = permissions

    def validate_api_key(self, api_key: str, required_permission: str) -> bool:
        """Validate if an API key exists and has the required permission"""
        if api_key not in self.api_keys:
            return False
        return required_permission in self.api_keys[api_key]

    def _generate_hmac(self, data: bytes) -> str:
        """Generate HMAC for data verification"""
        h = hmac.new(self.master_key.encode(), data, hashlib.sha256)
        return base64.b64encode(h.digest()).decode()

    def encode_tile_with_geo_privacy(
        self,
        image_data: np.ndarray,
        geometry: Any,
        metadata: Dict[str, Any],
        protection_level: str = "high",
        layout_type: str = "grid",
        fractal_type: Optional[str] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Encode tile data with geo-privacy protection"""
        # Convert numpy array to bytes
        img = Image.fromarray(image_data)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Create secure metadata
        secure_metadata = {
            "metadata": metadata,
            "geometry": mapping(geometry),
            "protection": {
                "level": protection_level,
                "layout": layout_type,
                "fractal": fractal_type
            },
            "hmac": self._generate_hmac(img_bytes)
        }

        return img_bytes, secure_metadata

    def decode_tile_with_geo_privacy(
        self,
        encrypted_data: bytes,
        secure_metadata: Dict[str, Any],
        access_token: str
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """Decode tile data and verify integrity"""
        # Verify HMAC
        if self._generate_hmac(encrypted_data) != secure_metadata.get("hmac"):
            return None, None

        # Convert bytes back to numpy array
        img = Image.open(io.BytesIO(encrypted_data))
        img_array = np.array(img)

        # Extract geometry
        geometry = shape(secure_metadata["geometry"])

        return img_array, geometry 