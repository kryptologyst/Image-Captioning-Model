#!/usr/bin/env python3
"""
Sample data generation and testing script.

This script generates sample images and tests the image captioning system
to ensure everything works end-to-end.
"""

import sys
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_generator import SyntheticImageGenerator
from image_captioner import ImageCaptioner
from advanced_captioner import AdvancedImageCaptioner, ModelType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """Generate sample images for testing."""
    print("🎨 Generating sample images...")
    
    generator = SyntheticImageGenerator()
    
    # Generate simple object images
    print("Creating object images...")
    dataset = generator.create_dataset(
        num_images=10,
        categories=["animals", "objects"]
    )
    
    # Generate scene images
    print("Creating scene images...")
    scene_paths = generator.create_scene_dataset(num_scenes=5)
    
    print(f"✅ Generated {sum(len(images) for images in dataset.values())} object images")
    print(f"✅ Generated {len(scene_paths)} scene images")
    
    return dataset, scene_paths


def test_basic_captioning():
    """Test basic image captioning functionality."""
    print("\n🖼️ Testing basic image captioning...")
    
    try:
        captioner = ImageCaptioner()
        
        # Find a sample image
        sample_dir = Path("data/synthetic")
        if not sample_dir.exists():
            print("❌ No sample images found. Run generate_sample_data() first.")
            return False
        
        image_files = list(sample_dir.glob("*.jpg"))
        if not image_files:
            print("❌ No image files found in sample directory.")
            return False
        
        test_image = image_files[0]
        print(f"Testing with image: {test_image.name}")
        
        # Generate caption
        caption = captioner.generate_caption(test_image)
        print(f"✅ Generated caption: {caption}")
        
        # Test multiple captions
        captions = captioner.generate_multiple_captions(test_image, num_captions=3)
        print(f"✅ Generated {len(captions)} diverse captions:")
        for i, cap in enumerate(captions, 1):
            print(f"  {i}. {cap}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic captioning test: {e}")
        return False


def test_advanced_captioning():
    """Test advanced captioning functionality."""
    print("\n🚀 Testing advanced captioning...")
    
    try:
        captioner = AdvancedImageCaptioner(
            models=[ModelType.BLIP_BASE],
            enable_ensemble=False,
            enable_clip_reranking=False
        )
        
        # Find a sample image
        sample_dir = Path("data/synthetic")
        image_files = list(sample_dir.glob("*.jpg"))
        
        if not image_files:
            print("❌ No sample images found.")
            return False
        
        test_image = image_files[0]
        print(f"Testing with image: {test_image.name}")
        
        # Test ensemble captioning
        result = captioner.generate_caption_ensemble(test_image)
        print(f"✅ Ensemble caption: {result.caption}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Time: {result.generation_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced captioning test: {e}")
        return False


def test_batch_processing():
    """Test batch processing functionality."""
    print("\n📊 Testing batch processing...")
    
    try:
        captioner = ImageCaptioner()
        
        # Get sample images
        sample_dir = Path("data/synthetic")
        image_files = list(sample_dir.glob("*.jpg"))[:3]  # Test with first 3 images
        
        if len(image_files) < 2:
            print("❌ Need at least 2 sample images for batch testing.")
            return False
        
        print(f"Processing {len(image_files)} images...")
        
        # Test batch processing
        results = captioner.batch_process(image_files)
        
        print(f"✅ Processed {len(results)} images:")
        for path, caption in results.items():
            print(f"  {Path(path).name}: {caption}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batch processing test: {e}")
        return False


def main():
    """Main testing function."""
    print("🧪 Image Captioning System Test Suite")
    print("=" * 50)
    
    # Generate sample data
    dataset, scene_paths = generate_sample_data()
    
    # Test basic functionality
    basic_success = test_basic_captioning()
    
    # Test advanced functionality
    advanced_success = test_advanced_captioning()
    
    # Test batch processing
    batch_success = test_batch_processing()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 20)
    print(f"Sample Data Generation: ✅")
    print(f"Basic Captioning: {'✅' if basic_success else '❌'}")
    print(f"Advanced Captioning: {'✅' if advanced_success else '❌'}")
    print(f"Batch Processing: {'✅' if batch_success else '❌'}")
    
    if all([basic_success, advanced_success, batch_success]):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the web interface: streamlit run web_app/app.py")
        print("2. Use the CLI: python cli.py caption <image_path>")
        print("3. Explore the advanced features in the web interface")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
