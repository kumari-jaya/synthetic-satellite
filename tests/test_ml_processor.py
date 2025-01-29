"""
Tests for the Advanced ML Processor
"""

import pytest
import numpy as np
import xarray as xr
import torch
import geopandas as gpd
from shapely.geometry import box
import cv2
from unittest.mock import Mock, patch

from tileformer.processors import AdvancedMLProcessor

@pytest.fixture
def processor():
    return AdvancedMLProcessor()

@pytest.fixture
def sample_data():
    # Create sample xarray data
    data = np.random.rand(3, 256, 256)
    return xr.DataArray(
        data,
        dims=('band', 'y', 'x'),
        coords={
            'band': ['R', 'G', 'B'],
            'y': range(256),
            'x': range(256)
        }
    )

class TestMLProcessor:
    """Test ML processor functionality"""
    
    @pytest.mark.asyncio
    async def test_device_setup(self, processor):
        """Test device setup and optimization"""
        assert isinstance(processor.device, torch.device)
        if torch.cuda.is_available():
            assert processor.device.type == 'cuda'
            assert 'tensorrt' in processor.optimizers or 'onnx' in processor.optimizers
    
    @pytest.mark.asyncio
    async def test_preprocessing(self, processor, sample_data):
        """Test advanced preprocessing pipeline"""
        # Test segmentation preprocessing
        processed = await processor._preprocess_advanced(sample_data, 'segmentation')
        assert isinstance(processed, xr.DataArray)
        assert processed.shape == sample_data.shape
        
        # Test classification preprocessing
        processed = await processor._preprocess_advanced(sample_data, 'classification')
        assert isinstance(processed, xr.DataArray)
        assert processed.shape == sample_data.shape
    
    @pytest.mark.asyncio
    async def test_sam_processing(self, processor, sample_data):
        """Test SAM model processing"""
        points = [[128, 128]]
        result = await processor.process_raster(
            data=sample_data,
            model_name='sam-vit-huge',
            task='sam',
            points=points,
            confidence_threshold=0.5
        )
        assert isinstance(result, (xr.DataArray, gpd.GeoDataFrame))
    
    @pytest.mark.asyncio
    async def test_diffusion_models(self, processor, sample_data):
        """Test diffusion model pipelines"""
        # Test SDXL
        result = await processor.process_raster(
            data=sample_data,
            model_name='sdxl',
            task='sdxl',
            prompt='test image',
            steps=2
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_data.shape
        
        # Test ControlNet
        result = await processor.process_raster(
            data=sample_data,
            model_name='controlnet',
            task='controlnet',
            prompt='test image',
            steps=2
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_data.shape
    
    @pytest.mark.asyncio
    async def test_optimization(self, processor, sample_data):
        """Test model optimization"""
        with patch('torch.cuda.is_available', return_value=True):
            # Test TensorRT optimization
            optimized = await processor._optimize_inference('sam-vit-huge', 'sam')
            assert optimized is not None
            
            # Test ONNX optimization
            optimized = await processor._optimize_inference('segformer-b0', 'segmentation')
            assert optimized is not None
    
    @pytest.mark.asyncio
    async def test_advanced_filters(self, processor, sample_data):
        """Test advanced filtering techniques"""
        # Test bilateral filter
        filtered = processor._bilateral_filter(sample_data)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == sample_data.shape
        
        # Test NLM denoising
        filtered = processor._nlm_denoise(sample_data)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == sample_data.shape
        
        # Test guided filter
        filtered = processor._guided_filter(sample_data)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == sample_data.shape
    
    @pytest.mark.asyncio
    async def test_encoders(self, processor):
        """Test encoding schemes"""
        # Test wavelet encoder
        wavelet = processor._create_wavelet_encoder()
        assert wavelet['transform'] == 'db4'
        assert wavelet['level'] == 3
        
        # Test Fourier encoder
        fourier = processor._create_fourier_encoder()
        assert fourier['type'] == 'fft2'
        assert fourier['norm'] == 'ortho'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor, sample_data):
        """Test error handling"""
        # Test invalid model
        with pytest.raises(ValueError):
            await processor.process_raster(
                data=sample_data,
                model_name='invalid_model',
                task='segmentation'
            )
        
        # Test invalid task
        with pytest.raises(ValueError):
            await processor.process_raster(
                data=sample_data,
                model_name='sam-vit-huge',
                task='invalid_task'
            )
    
    @pytest.mark.asyncio
    async def test_performance(self, processor, sample_data):
        """Test performance metrics"""
        import time
        
        # Measure processing time
        start_time = time.time()
        await processor.process_raster(
            data=sample_data,
            model_name='sam-vit-huge',
            task='sam',
            optimization_level='high'
        )
        processing_time = time.time() - start_time
        
        # Check performance meets requirements
        assert processing_time < 2.0  # Should process in under 2 seconds
        
        if torch.cuda.is_available():
            # Check GPU memory usage
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            assert gpu_mem < 2048  # Should use less than 2GB GPU memory
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """Test batch processing capabilities"""
        # Create batch of images
        batch_size = 4
        batch_data = [np.random.rand(3, 256, 256) for _ in range(batch_size)]
        batch_arrays = [
            xr.DataArray(
                data,
                dims=('band', 'y', 'x'),
                coords={
                    'band': ['R', 'G', 'B'],
                    'y': range(256),
                    'x': range(256)
                }
            )
            for data in batch_data
        ]
        
        # Process batch
        results = await asyncio.gather(*[
            processor.process_raster(
                data=arr,
                model_name='sam-vit-huge',
                task='sam'
            )
            for arr in batch_arrays
        ])
        
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, (xr.DataArray, gpd.GeoDataFrame))

if __name__ == '__main__':
    pytest.main([__file__]) 