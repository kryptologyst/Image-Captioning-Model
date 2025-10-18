"""
Data generation module for creating synthetic datasets.

This module provides utilities to generate synthetic images and datasets
for testing and demonstration purposes.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)


class SyntheticImageGenerator:
    """Generate synthetic images for testing image captioning models."""
    
    def __init__(self, output_dir: str = "data/synthetic"):
        """
        Initialize the synthetic image generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define image categories and their characteristics
        self.categories = {
            "animals": {
                "shapes": ["circle", "rectangle", "ellipse"],
                "colors": ["red", "blue", "green", "yellow", "orange", "purple"],
                "objects": ["cat", "dog", "bird", "fish", "rabbit", "elephant"]
            },
            "vehicles": {
                "shapes": ["rectangle", "ellipse"],
                "colors": ["red", "blue", "black", "white", "gray"],
                "objects": ["car", "truck", "bus", "motorcycle", "bicycle"]
            },
            "nature": {
                "shapes": ["circle", "ellipse", "polygon"],
                "colors": ["green", "brown", "blue", "yellow"],
                "objects": ["tree", "flower", "mountain", "river", "sun"]
            },
            "objects": {
                "shapes": ["rectangle", "circle", "square"],
                "colors": ["red", "blue", "green", "yellow", "black", "white"],
                "objects": ["ball", "box", "book", "phone", "laptop"]
            }
        }
    
    def generate_simple_image(
        self, 
        category: str,
        width: int = 224,
        height: int = 224,
        background_color: str = "white"
    ) -> Image.Image:
        """
        Generate a simple synthetic image.
        
        Args:
            category: Image category (animals, vehicles, nature, objects)
            width: Image width
            height: Image height
            background_color: Background color
            
        Returns:
            Generated PIL Image
        """
        # Create blank image
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)
        
        if category not in self.categories:
            category = "objects"
        
        cat_data = self.categories[category]
        
        # Generate random object
        obj_name = random.choice(cat_data["objects"])
        color = random.choice(cat_data["colors"])
        shape = random.choice(cat_data["shapes"])
        
        # Draw the object
        self._draw_object(draw, shape, color, width, height)
        
        return img
    
    def _draw_object(
        self, 
        draw: ImageDraw.Draw, 
        shape: str, 
        color: str, 
        width: int, 
        height: int
    ) -> None:
        """Draw a simple geometric object."""
        # Define color mapping
        color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "brown": (165, 42, 42)
        }
        
        rgb_color = color_map.get(color, (0, 0, 0))
        
        # Calculate object position and size
        obj_width = random.randint(30, min(width, height) // 2)
        obj_height = random.randint(30, min(width, height) // 2)
        
        x = random.randint(0, width - obj_width)
        y = random.randint(0, height - obj_height)
        
        # Draw shape
        if shape == "circle":
            draw.ellipse([x, y, x + obj_width, y + obj_height], fill=rgb_color)
        elif shape == "rectangle":
            draw.rectangle([x, y, x + obj_width, y + obj_height], fill=rgb_color)
        elif shape == "square":
            size = min(obj_width, obj_height)
            draw.rectangle([x, y, x + size, y + size], fill=rgb_color)
        elif shape == "ellipse":
            draw.ellipse([x, y, x + obj_width, y + obj_height], fill=rgb_color)
    
    def generate_scene_image(
        self, 
        scene_type: str = "park",
        width: int = 224,
        height: int = 224
    ) -> Image.Image:
        """
        Generate a more complex scene image.
        
        Args:
            scene_type: Type of scene to generate
            width: Image width
            height: Image height
            
        Returns:
            Generated PIL Image
        """
        img = Image.new('RGB', (width, height), (135, 206, 235))  # Sky blue
        draw = ImageDraw.Draw(img)
        
        if scene_type == "park":
            self._draw_park_scene(draw, width, height)
        elif scene_type == "street":
            self._draw_street_scene(draw, width, height)
        elif scene_type == "room":
            self._draw_room_scene(draw, width, height)
        else:
            self._draw_simple_scene(draw, width, height)
        
        return img
    
    def _draw_park_scene(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a park scene."""
        # Ground
        draw.rectangle([0, height * 2 // 3, width, height], fill=(34, 139, 34))
        
        # Trees
        for _ in range(3):
            x = random.randint(0, width - 20)
            y = height * 2 // 3 - 40
            # Tree trunk
            draw.rectangle([x + 8, y + 20, x + 12, y + 40], fill=(139, 69, 19))
            # Tree top
            draw.ellipse([x, y, x + 20, y + 30], fill=(0, 100, 0))
        
        # Sun
        draw.ellipse([width - 40, 10, width - 10, 40], fill=(255, 255, 0))
    
    def _draw_street_scene(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a street scene."""
        # Road
        draw.rectangle([0, height * 2 // 3, width, height], fill=(64, 64, 64))
        
        # Buildings
        for _ in range(2):
            x = random.randint(0, width - 60)
            y = random.randint(0, height // 2)
            draw.rectangle([x, y, x + 60, height * 2 // 3], fill=(105, 105, 105))
    
    def _draw_room_scene(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a room scene."""
        # Floor
        draw.rectangle([0, height * 2 // 3, width, height], fill=(222, 184, 135))
        
        # Wall
        draw.rectangle([0, 0, width, height * 2 // 3], fill=(255, 228, 196))
        
        # Furniture
        draw.rectangle([20, height // 2, 80, height * 2 // 3], fill=(139, 69, 19))
    
    def _draw_simple_scene(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a simple scene."""
        # Add some random shapes
        for _ in range(3):
            shape = random.choice(["circle", "rectangle"])
            color = random.choice(["red", "blue", "green", "yellow"])
            self._draw_object(draw, shape, color, width, height)
    
    def create_dataset(
        self, 
        num_images: int = 50,
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Create a synthetic dataset.
        
        Args:
            num_images: Number of images to generate
            categories: List of categories to include
            
        Returns:
            Dictionary mapping categories to image paths
        """
        if categories is None:
            categories = list(self.categories.keys())
        
        dataset = {category: [] for category in categories}
        
        images_per_category = num_images // len(categories)
        
        for category in categories:
            for i in range(images_per_category):
                # Generate image
                img = self.generate_simple_image(category)
                
                # Save image
                filename = f"{category}_{i:03d}.jpg"
                filepath = self.output_dir / filename
                img.save(filepath)
                
                dataset[category].append(str(filepath))
                
                logger.info(f"Generated {filename}")
        
        # Save dataset metadata
        metadata = {
            "total_images": num_images,
            "categories": categories,
            "images_per_category": images_per_category,
            "dataset": dataset
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset created with {num_images} images")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return dataset
    
    def create_scene_dataset(self, num_scenes: int = 20) -> List[str]:
        """
        Create a dataset of scene images.
        
        Args:
            num_scenes: Number of scene images to generate
            
        Returns:
            List of image file paths
        """
        scene_types = ["park", "street", "room"]
        image_paths = []
        
        for i in range(num_scenes):
            scene_type = random.choice(scene_types)
            img = self.generate_scene_image(scene_type)
            
            filename = f"scene_{scene_type}_{i:03d}.jpg"
            filepath = self.output_dir / filename
            img.save(filepath)
            
            image_paths.append(str(filepath))
            logger.info(f"Generated scene: {filename}")
        
        return image_paths


def create_sample_images() -> None:
    """Create sample images for testing."""
    generator = SyntheticImageGenerator()
    
    # Create simple object dataset
    print("Creating simple object dataset...")
    dataset = generator.create_dataset(num_images=20)
    
    # Create scene dataset
    print("Creating scene dataset...")
    scene_paths = generator.create_scene_dataset(num_scenes=10)
    
    print(f"Sample images created in {generator.output_dir}")
    print("You can now test the image captioning model with these images!")


if __name__ == "__main__":
    create_sample_images()
