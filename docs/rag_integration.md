# Integrating Earth Memories with RAG Systems

This guide explains how to integrate Vortx's Earth Memory system with Retrieval-Augmented Generation (RAG) systems to provide rich contextual information about Earth locations and their temporal evolution.

## Overview

The Earth Memory system provides:
1. Dense vector embeddings of Earth observation data
2. Temporal understanding of location changes
3. Multi-modal data synthesis
4. Fast similarity search
5. Rich contextual metadata

## Integration Methods

### 1. Direct Memory Access

Use the memory store directly as a vector database:

```python
from vortx.core.memory import EarthMemoryStore
from pathlib import Path

# Initialize memory store
memory_store = EarthMemoryStore(
    base_path=Path("./data/memories"),
    index_type="faiss"  # For fast similarity search
)

# Query memories
memories = memory_store.query_memories(
    coordinates=(37.7749, -122.4194),  # San Francisco
    time_range=(start_time, end_time),
    k=5
)

# Use in RAG context
context = []
for memory in memories:
    context.append({
        "content": f"Location: {memory['coordinates']}\n"
                  f"Time: {memory['timestamp']}\n"
                  f"Data: {memory['metadata']}",
        "embedding": memory["embedding"]
    })
```

### 2. REST API Integration

Use the provided REST API endpoints:

```python
import requests
import json

# Query location context
response = requests.post(
    "http://api.vortx.com/analysis/get_context",
    json={
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": "2024-03-15T00:00:00Z",
        "context_window_days": 365
    }
)

context = response.json()

# Use in RAG prompt
prompt = f"""
Answer based on the following Earth observation context:
Location: {context['current_state']['coordinates']}
Time: {context['current_state']['timestamp']}
Historical Trends: {json.dumps(context['historical_trends'], indent=2)}
"""
```

### 3. LangChain Integration

Use the Vortx memory store as a LangChain retriever:

```python
from langchain.retrievers import VortxRetriever
from langchain.chains import RetrievalQA

# Initialize retriever
retriever = VortxRetriever(
    memory_store=memory_store,
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    chain_type="stuff",
    retriever=retriever
)

# Query with location context
response = qa_chain.run(
    "What environmental changes have occurred in San Francisco over the past year?"
)
```

### 4. Custom Embedding Pipeline

Create a custom embedding pipeline for your specific use case:

```python
from vortx.core.synthesis import SynthesisPipeline
from vortx.core.data_sources import (
    SatelliteDataSource,
    WeatherDataSource,
    ClimateDataSource
)

# Configure data sources
data_sources = [
    SatelliteDataSource(...),
    WeatherDataSource(...),
    ClimateDataSource(...)
]

# Create pipeline
pipeline = SynthesisPipeline(
    data_sources=data_sources,
    memory_store=memory_store
)

# Process location
result = pipeline.process_location(
    coordinates=(37.7749, -122.4194),
    timestamp=current_time,
    metadata={"location_name": "San Francisco"}
)

# Use embeddings in your RAG system
embedding = result["embedding"]
```

## Best Practices

1. **Temporal Context**
   - Always include temporal information in queries
   - Consider seasonal and cyclical patterns
   - Use appropriate time windows for different phenomena

2. **Spatial Context**
   - Consider the spatial resolution of different data sources
   - Use appropriate search radii for different queries
   - Account for geographic relationships

3. **Data Synthesis**
   - Combine multiple data sources for richer context
   - Use attention maps to focus on relevant features
   - Consider data source reliability and resolution

4. **Performance Optimization**
   - Use FAISS for large-scale similarity search
   - Cache frequently accessed memories
   - Batch process locations when possible

## Example: Environmental Impact Assessment

```python
# Query environmental changes
changes = requests.post(
    "http://api.vortx.com/analysis/detect_changes",
    json={
        "latitude": 37.7749,
        "longitude": -122.4194,
        "start_time": "2023-03-15T00:00:00Z",
        "end_time": "2024-03-15T00:00:00Z",
        "interval_days": 30,
        "change_threshold": 0.1
    }
).json()

# Create RAG prompt with environmental context
prompt = f"""
Analyze the environmental impact based on the following changes:
Location: San Francisco (37.7749°N, 122.4194°W)
Time Period: 2023-03-15 to 2024-03-15
Significant Changes:
{json.dumps(changes['significant_changes'], indent=2)}

Question: What are the major environmental trends and their potential impacts?
"""
```

## Advanced Features

1. **Attention Visualization**
   ```python
   # Get embeddings with attention maps
   result = pipeline.process_location(...)
   attention_maps = result["attention_maps"]
   
   # Visualize spatial attention
   import matplotlib.pyplot as plt
   plt.imshow(attention_maps["spatial"][0, 0].numpy())
   plt.title("Spatial Attention Map")
   plt.colorbar()
   plt.show()
   ```

2. **Custom Similarity Metrics**
   ```python
   def custom_similarity(embedding1, embedding2, weights=None):
       if weights is None:
           weights = torch.ones_like(embedding1)
       return torch.sum(weights * (embedding1 * embedding2))
   ```

3. **Temporal Aggregation**
   ```python
   def aggregate_temporal_embeddings(embeddings, timestamps, window_size):
       """Aggregate embeddings within time windows."""
       # Implementation
       pass
   ```

## Troubleshooting

1. **Memory Usage**
   - Use streaming for large datasets
   - Implement memory cleanup strategies
   - Monitor FAISS index size

2. **Query Performance**
   - Use appropriate FAISS index type
   - Implement caching
   - Optimize batch sizes

3. **Data Quality**
   - Handle missing data gracefully
   - Validate data source reliability
   - Implement data quality checks

## Future Developments

1. **Enhanced Features**
   - Multi-scale temporal analysis
   - Cross-location relationship modeling
   - Causal inference support

2. **Integration Plans**
   - Additional LLM framework support
   - Real-time data processing
   - Advanced visualization tools 