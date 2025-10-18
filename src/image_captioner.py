"""
Image Captioning Module

This module provides a modern, type-safe implementation of image captioning
using state-of-the-art BLIP models from Hugging Face Transformers.

Author: AI Assistant
Date: 2024
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings

import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    pipeline,
    Pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptioner:
    """
    A modern image captioning class using BLIP models.
    
    This class provides a clean interface for generating captions from images
    using pre-trained BLIP (Bootstrapped Language Image Pretraining) models.
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
        use_pipeline: bool = True
    ) -> None:
        """
        Initialize the ImageCaptioner.
        
        Args:
            model_name: Hugging Face model identifier for BLIP
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            use_pipeline: Whether to use Hugging Face pipeline (recommended)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_pipeline = use_pipeline
        
        logger.info(f"Initializing ImageCaptioner with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the BLIP model and processor."""
        try:
            if self.use_pipeline:
                # Use Hugging Face pipeline for cleaner interface
                self.pipeline = pipeline(
                    "image-to-text",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Model loaded successfully using pipeline")
            else:
                # Load model and processor separately for more control
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name
                ).to(self.device)
                logger.info("Model and processor loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be opened
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Successfully loaded image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Cannot open image: {e}")
    
    def generate_caption(
        self, 
        image: Union[str, Path, Image.Image],
        max_length: int = 50,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image: Image path, Path object, or PIL Image
            max_length: Maximum length of generated caption
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated caption string
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        try:
            if self.use_pipeline:
                # Use pipeline for generation
                result = self.pipeline(
                    image,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=do_sample
                )
                caption = result[0]['generated_text']
            else:
                # Use model directly
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        do_sample=do_sample
                    )
                
                caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            logger.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            raise
    
    def generate_multiple_captions(
        self, 
        image: Union[str, Path, Image.Image],
        num_captions: int = 3,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple diverse captions for an image.
        
        Args:
            image: Image path, Path object, or PIL Image
            num_captions: Number of captions to generate
            **kwargs: Additional arguments for generate_caption
            
        Returns:
            List of generated captions
        """
        captions = []
        
        for i in range(num_captions):
            # Vary parameters for diversity
            kwargs_copy = kwargs.copy()
            kwargs_copy['temperature'] = kwargs_copy.get('temperature', 1.0) + i * 0.2
            kwargs_copy['do_sample'] = True
            
            caption = self.generate_caption(image, **kwargs_copy)
            captions.append(caption)
        
        return captions
    
    def visualize_result(
        self, 
        image: Union[str, Path, Image.Image],
        caption: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Visualize the image with its caption.
        
        Args:
            image: Image path, Path object, or PIL Image
            caption: Caption to display (generates if None)
            save_path: Path to save the visualization
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Generate caption if not provided
        if caption is None:
            caption = self.generate_caption(image)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f"Caption: {caption}", fontsize=14, wrap=True)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def batch_process(
        self, 
        image_paths: List[Union[str, Path]],
        **kwargs
    ) -> Dict[str, str]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            **kwargs: Arguments for generate_caption
            
        Returns:
            Dictionary mapping image paths to captions
        """
        results = {}
        
        for image_path in image_paths:
            try:
                caption = self.generate_caption(image_path, **kwargs)
                results[str(image_path)] = caption
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[str(image_path)] = f"Error: {e}"
        
        return results


def main() -> None:
    """Example usage of the ImageCaptioner class."""
    # Initialize captioner
    captioner = ImageCaptioner()
    
    # Example with a sample image (you'll need to provide an actual image)
    try:
        # This will work if you have an image file
        image_path = "data/sample_image.jpg"
        caption = captioner.generate_caption(image_path)
        print(f"Generated caption: {caption}")
        
        # Visualize result
        captioner.visualize_result(image_path)
        
    except FileNotFoundError:
        print("No sample image found. Please add an image to test the captioner.")
        print("You can use the create_sample_data.py script to generate sample images.")


if __name__ == "__main__":
    main()
