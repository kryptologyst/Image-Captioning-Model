"""
Test suite for the Image Captioning project.

This module contains unit tests for all major components.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_captioner import ImageCaptioner
from config import Config, ModelConfig, AppConfig
from data_generator import SyntheticImageGenerator


class TestImageCaptioner:
    """Test cases for ImageCaptioner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        
        # Create a simple test image
        from PIL import Image
        test_img = Image.new('RGB', (224, 224), color='red')
        test_img.save(self.test_image_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.image_captioner.BlipProcessor')
    @patch('src.image_captioner.BlipForConditionalGeneration')
    def test_init_with_pipeline(self, mock_model, mock_processor):
        """Test ImageCaptioner initialization with pipeline."""
        # Mock the pipeline
        with patch('src.image_captioner.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            captioner = ImageCaptioner(use_pipeline=True)
            
            assert captioner.use_pipeline is True
            assert captioner.device in ["cpu", "cuda", "mps"]
            mock_pipeline.assert_called_once()
    
    @patch('src.image_captioner.BlipProcessor')
    @patch('src.image_captioner.BlipForConditionalGeneration')
    def test_init_without_pipeline(self, mock_model, mock_processor):
        """Test ImageCaptioner initialization without pipeline."""
        mock_processor.return_value = Mock()
        mock_model.return_value = Mock()
        
        captioner = ImageCaptioner(use_pipeline=False)
        
        assert captioner.use_pipeline is False
        mock_processor.assert_called_once()
        mock_model.assert_called_once()
    
    def test_load_image_valid_path(self):
        """Test loading a valid image."""
        with patch('src.image_captioner.pipeline'):
            captioner = ImageCaptioner()
            image = captioner.load_image(self.test_image_path)
            
            assert image is not None
            assert hasattr(image, 'size')
    
    def test_load_image_invalid_path(self):
        """Test loading an invalid image path."""
        with patch('src.image_captioner.pipeline'):
            captioner = ImageCaptioner()
            
            with pytest.raises(FileNotFoundError):
                captioner.load_image("nonexistent_image.jpg")
    
    @patch('src.image_captioner.pipeline')
    def test_generate_caption_with_pipeline(self, mock_pipeline):
        """Test caption generation with pipeline."""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': 'test caption'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        captioner = ImageCaptioner(use_pipeline=True)
        caption = captioner.generate_caption(self.test_image_path)
        
        assert caption == 'test caption'
        mock_pipeline_instance.assert_called_once()
    
    @patch('src.image_captioner.BlipProcessor')
    @patch('src.image_captioner.BlipForConditionalGeneration')
    def test_generate_caption_without_pipeline(self, mock_model, mock_processor):
        """Test caption generation without pipeline."""
        mock_processor_instance = Mock()
        mock_processor_instance.decode.return_value = 'test caption'
        mock_processor_instance.return_value = {'input_ids': Mock()}
        mock_processor.return_value = mock_processor_instance
        
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = Mock()
        mock_model.return_value = mock_model_instance
        
        captioner = ImageCaptioner(use_pipeline=False)
        caption = captioner.generate_caption(self.test_image_path)
        
        assert caption == 'test caption'
        mock_processor_instance.assert_called()
        mock_model_instance.generate.assert_called_once()
    
    @patch('src.image_captioner.pipeline')
    def test_generate_multiple_captions(self, mock_pipeline):
        """Test generating multiple captions."""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': f'test caption {i}'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        captioner = ImageCaptioner(use_pipeline=True)
        captions = captioner.generate_multiple_captions(self.test_image_path, num_captions=3)
        
        assert len(captions) == 3
        assert all(isinstance(caption, str) for caption in captions)
    
    @patch('src.image_captioner.pipeline')
    @patch('src.image_captioner.plt')
    def test_visualize_result(self, mock_plt, mock_pipeline):
        """Test visualization functionality."""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': 'test caption'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        captioner = ImageCaptioner(use_pipeline=True)
        
        # Test with provided caption
        captioner.visualize_result(self.test_image_path, caption="provided caption")
        
        # Test without caption (should generate one)
        captioner.visualize_result(self.test_image_path)
        
        # Verify matplotlib was called
        assert mock_plt.figure.called
        assert mock_plt.imshow.called
        assert mock_plt.title.called
        assert mock_plt.show.called


class TestConfig:
    """Test cases for Config class."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        assert config.model_name == "Salesforce/blip-image-captioning-base"
        assert config.device is None
        assert config.use_pipeline is True
        assert config.max_length == 50
        assert config.num_beams == 4
        assert config.temperature == 1.0
        assert config.do_sample is False
    
    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.output_dir == "output"
    
    def test_config_init(self):
        """Test Config initialization."""
        config = Config()
        
        assert isinstance(config.model_config, ModelConfig)
        assert isinstance(config.app_config, AppConfig)
    
    def test_get_model_kwargs(self):
        """Test getting model kwargs."""
        config = Config()
        kwargs = config.get_model_kwargs()
        
        assert 'model_name' in kwargs
        assert 'device' in kwargs
        assert 'use_pipeline' in kwargs
    
    def test_get_generation_kwargs(self):
        """Test getting generation kwargs."""
        config = Config()
        kwargs = config.get_generation_kwargs()
        
        assert 'max_length' in kwargs
        assert 'num_beams' in kwargs
        assert 'temperature' in kwargs
        assert 'do_sample' in kwargs


class TestSyntheticImageGenerator:
    """Test cases for SyntheticImageGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SyntheticImageGenerator(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test SyntheticImageGenerator initialization."""
        assert self.generator.output_dir == Path(self.temp_dir)
        assert 'animals' in self.generator.categories
        assert 'vehicles' in self.generator.categories
        assert 'nature' in self.generator.categories
        assert 'objects' in self.generator.categories
    
    def test_generate_simple_image(self):
        """Test generating a simple image."""
        image = self.generator.generate_simple_image("animals")
        
        assert image is not None
        assert hasattr(image, 'size')
        assert image.size == (224, 224)
    
    def test_generate_scene_image(self):
        """Test generating a scene image."""
        image = self.generator.generate_scene_image("park")
        
        assert image is not None
        assert hasattr(image, 'size')
        assert image.size == (224, 224)
    
    def test_create_dataset(self):
        """Test creating a dataset."""
        dataset = self.generator.create_dataset(num_images=4, categories=["animals"])
        
        assert "animals" in dataset
        assert len(dataset["animals"]) == 4
        
        # Check that files were created
        for image_path in dataset["animals"]:
            assert Path(image_path).exists()
    
    def test_create_scene_dataset(self):
        """Test creating a scene dataset."""
        scene_paths = self.generator.create_scene_dataset(num_scenes=3)
        
        assert len(scene_paths) == 3
        
        # Check that files were created
        for image_path in scene_paths:
            assert Path(image_path).exists()


class TestIntegration:
    """Integration tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test image
        from PIL import Image
        test_img = Image.new('RGB', (224, 224), color='blue')
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        test_img.save(self.test_image_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.image_captioner.pipeline')
    def test_end_to_end_captioning(self, mock_pipeline):
        """Test end-to-end captioning workflow."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'generated_text': 'a blue image'}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Initialize captioner
        captioner = ImageCaptioner()
        
        # Generate caption
        caption = captioner.generate_caption(self.test_image_path)
        
        assert caption == 'a blue image'
        mock_pipeline_instance.assert_called_once()
    
    def test_config_and_captioner_integration(self):
        """Test integration between Config and ImageCaptioner."""
        config = Config()
        
        with patch('src.image_captioner.pipeline'):
            captioner = ImageCaptioner(**config.get_model_kwargs())
            
            assert captioner.model_name == config.model_config.model_name
            assert captioner.use_pipeline == config.model_config.use_pipeline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
