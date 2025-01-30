from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch import nn
import rasterio
from datetime import datetime
from pathlib import Path

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (attended features, attention weights)
        """
        attention = self.conv(x)
        return x * attention, attention

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for focusing on important time steps."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            Tuple of (attended features, attention weights)
        """
        attention = self.attention(x)
        return x * attention, attention

class EarthMemoryEncoder(nn.Module):
    """Enhanced encoder for converting earth observation data into memory embeddings."""
    
    def __init__(
        self,
        input_channels: int = 12,
        embedding_dim: int = 512,
        temporal_window: int = 24,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.temporal_window = temporal_window
        self.use_attention = use_attention
        
        # Spatial encoder with residual connections
        self.spatial_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels if i == 0 else 256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout2d(dropout)
            )
            for i in range(3)
        ])
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(256) if use_attention else None
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=256,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(embedding_dim) if use_attention else None
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, temporal_window, channels, height, width)
        
        Returns:
            Tuple of (embeddings, attention_maps)
        """
        batch_size = x.size(0)
        attention_maps = {}
        
        # Reshape for spatial encoding
        x = x.view(-1, self.input_channels, x.size(-2), x.size(-1))
        
        # Apply spatial encoder with residual connections
        spatial_features = x
        for layer in self.spatial_encoder:
            spatial_features = layer(spatial_features) + spatial_features
            
        # Apply spatial attention if enabled
        if self.use_attention and self.spatial_attention is not None:
            spatial_features, spatial_attn = self.spatial_attention(spatial_features)
            attention_maps["spatial"] = spatial_attn
            
        # Global pooling
        spatial_features = self.global_pool(spatial_features)
        spatial_features = spatial_features.view(batch_size, self.temporal_window, -1)
        
        # Apply temporal encoder
        temporal_features, (hidden, cell) = self.temporal_encoder(spatial_features)
        
        # Apply temporal attention if enabled
        if self.use_attention and self.temporal_attention is not None:
            temporal_features, temporal_attn = self.temporal_attention(temporal_features)
            attention_maps["temporal"] = temporal_attn
            
        # Get final embedding
        embedding = temporal_features[:, -1, :]
        
        # Apply final projection and normalization
        embedding = self.projection(embedding)
        embedding = self.layer_norm(embedding)
        
        return embedding, attention_maps

class EarthMemoryStore:
    """Enhanced storage and retrieval system for earth observation memories."""
    
    def __init__(
        self,
        base_path: Path,
        encoder: Optional[EarthMemoryEncoder] = None,
        index_type: str = "faiss"
    ):
        self.base_path = Path(base_path)
        self.encoder = encoder or EarthMemoryEncoder()
        self.index_type = index_type
        self.memory_index = {}
        
        # Initialize FAISS index if specified
        if index_type == "faiss":
            import faiss
            self.vector_index = faiss.IndexFlatL2(self.encoder.embedding_dim)
            self.id_to_key = []
        
    def encode_observation(
        self,
        data_path: Path,
        timestamp: datetime,
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode a single earth observation into a memory embedding.
        
        Args:
            data_path: Path to the observation data
            timestamp: Timestamp of the observation
            metadata: Additional metadata about the observation
            
        Returns:
            Tuple of (embedding tensor, attention maps)
        """
        with rasterio.open(data_path) as src:
            data = src.read()
            data = torch.from_numpy(data).float()
            data = data.unsqueeze(0).unsqueeze(0)  # Add batch and temporal dims
            
        with torch.no_grad():
            embedding, attention_maps = self.encoder(data)
            
        return embedding, attention_maps
    
    def store_memory(
        self,
        embedding: torch.Tensor,
        coordinates: tuple,
        timestamp: datetime,
        metadata: Dict[str, Any],
        attention_maps: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Store a memory embedding with its metadata and attention maps."""
        memory_key = f"{coordinates[0]}_{coordinates[1]}_{timestamp.isoformat()}"
        
        memory_data = {
            "embedding": embedding,
            "coordinates": coordinates,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        if attention_maps:
            memory_data["attention_maps"] = attention_maps
            
        self.memory_index[memory_key] = memory_data
        
        # Update FAISS index if used
        if self.index_type == "faiss":
            self.vector_index.add(embedding.numpy().reshape(1, -1))
            self.id_to_key.append(memory_key)
    
    def query_memories(
        self,
        coordinates: tuple,
        time_range: Optional[tuple] = None,
        k: int = 5,
        use_faiss: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query memories by location and optional time range.
        
        Args:
            coordinates: (latitude, longitude) tuple
            time_range: Optional (start_time, end_time) tuple
            k: Number of memories to return
            use_faiss: Whether to use FAISS for similarity search
            
        Returns:
            List of memory records
        """
        if use_faiss and self.index_type == "faiss":
            # Get reference embedding for the query location
            reference = next(iter(self.memory_index.values()))["embedding"]
            
            # Search using FAISS
            distances, indices = self.vector_index.search(
                reference.numpy().reshape(1, -1),
                k
            )
            
            results = []
            for i, idx in enumerate(indices[0]):
                memory_key = self.id_to_key[idx]
                memory = self.memory_index[memory_key]
                
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= memory["timestamp"] <= end_time):
                        continue
                        
                results.append(memory)
                
            return results
        else:
            # Fallback to original distance-based search
            results = []
            for key, memory in self.memory_index.items():
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= memory["timestamp"] <= end_time):
                        continue
                        
                dist = np.sqrt(
                    (memory["coordinates"][0] - coordinates[0])**2 +
                    (memory["coordinates"][1] - coordinates[1])**2
                )
                
                results.append((dist, memory))
                
            results.sort(key=lambda x: x[0])
            return [memory for _, memory in results[:k]]
    
    def save_index(self, path: Optional[Path] = None):
        """Save the memory index and FAISS index to disk."""
        save_path = path or self.base_path
        
        # Save memory index
        torch.save(self.memory_index, save_path / "memory_index.pt")
        
        # Save FAISS index if used
        if self.index_type == "faiss":
            import faiss
            faiss.write_index(
                self.vector_index,
                str(save_path / "vector_index.faiss")
            )
            torch.save(self.id_to_key, save_path / "id_to_key.pt")
    
    def load_index(self, path: Optional[Path] = None):
        """Load the memory index and FAISS index from disk."""
        load_path = path or self.base_path
        
        # Load memory index
        self.memory_index = torch.load(load_path / "memory_index.pt")
        
        # Load FAISS index if used
        if self.index_type == "faiss":
            import faiss
            self.vector_index = faiss.read_index(
                str(load_path / "vector_index.faiss")
            )
            self.id_to_key = torch.load(load_path / "id_to_key.pt") 