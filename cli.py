"""
Command Line Interface for Image Captioning

This module provides a comprehensive CLI for the image captioning system.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_captioner import ImageCaptioner
from advanced_captioner import AdvancedImageCaptioner, ModelType
from config import Config
from data_generator import SyntheticImageGenerator


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_single_image(args) -> None:
    """Handle single image captioning command."""
    captioner = ImageCaptioner(
        model_name=args.model,
        device=args.device,
        use_pipeline=args.use_pipeline
    )
    
    try:
        caption = captioner.generate_caption(
            args.image,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
        
        print(f"Caption: {caption}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"caption": caption, "image": str(args.image)}, f, indent=2)
            print(f"Result saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_batch_process(args) -> None:
    """Handle batch processing command."""
    captioner = ImageCaptioner(
        model_name=args.model,
        device=args.device,
        use_pipeline=args.use_pipeline
    )
    
    # Get list of image files
    image_dir = Path(args.input_dir)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        print(f"Error: No image files found in {image_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(image_files)} images...")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            caption = captioner.generate_caption(
                image_file,
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=args.temperature,
                do_sample=args.do_sample
            )
            
            results.append({
                "image": str(image_file),
                "caption": caption,
                "status": "success"
            })
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            results.append({
                "image": str(image_file),
                "caption": "",
                "status": "error",
                "error": str(e)
            })
    
    # Save results
    output_file = args.output or "batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"Successfully processed: {successful}/{len(results)} images")


def cmd_generate_samples(args) -> None:
    """Handle sample generation command."""
    generator = SyntheticImageGenerator(output_dir=args.output_dir)
    
    if args.scenes:
        print("Generating scene images...")
        scene_paths = generator.create_scene_dataset(num_scenes=args.num_images)
        print(f"Generated {len(scene_paths)} scene images")
    else:
        print("Generating object images...")
        dataset = generator.create_dataset(
            num_images=args.num_images,
            categories=args.categories
        )
        
        total_images = sum(len(images) for images in dataset.values())
        print(f"Generated {total_images} images across {len(dataset)} categories")


def cmd_advanced_captioning(args) -> None:
    """Handle advanced captioning commands."""
    # Parse model types
    model_types = []
    for model_name in args.models:
        try:
            model_type = ModelType(model_name)
            model_types.append(model_type)
        except ValueError:
            print(f"Error: Unknown model {model_name}", file=sys.stderr)
            sys.exit(1)
    
    captioner = AdvancedImageCaptioner(
        models=model_types,
        device=args.device,
        enable_ensemble=args.enable_ensemble,
        enable_clip_reranking=args.enable_clip_reranking
    )
    
    try:
        if args.method == "ensemble":
            result = captioner.generate_caption_ensemble(
                args.image,
                num_captions_per_model=args.num_captions,
                ensemble_strategy=args.strategy
            )
        
        elif args.method == "zero_shot":
            result = captioner.zero_shot_captioning(args.image)
        
        elif args.method == "compare":
            results = captioner.compare_models(args.image, args.methods)
            
            print("=== Method Comparison ===")
            for method, result in results.items():
                print(f"{method}:")
                print(f"  Caption: {result.caption}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Time: {result.generation_time:.2f}s")
                print()
            
            return
        
        else:
            print(f"Error: Unknown method {args.method}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Caption: {result.caption}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Generation Time: {result.generation_time:.2f}s")
        print(f"Model: {result.model_name}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "caption": result.caption,
                    "confidence": result.confidence,
                    "model_name": result.model_name,
                    "generation_time": result.generation_time,
                    "metadata": result.metadata
                }, f, indent=2)
            print(f"Result saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_web_interface(args) -> None:
    """Handle web interface command."""
    import subprocess
    
    app_path = Path(__file__).parent.parent / "web_app" / "app.py"
    
    cmd = [
        "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    print(f"Starting web interface on http://{args.host}:{args.port}")
    subprocess.run(cmd)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Image Captioning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Caption a single image
  python cli.py caption image.jpg

  # Batch process images
  python cli.py batch input_dir/ --output results.json

  # Generate sample images
  python cli.py generate-samples --num-images 20

  # Advanced ensemble captioning
  python cli.py advanced image.jpg --method ensemble --models blip-base clip

  # Start web interface
  python cli.py web --port 8501
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single image captioning
    caption_parser = subparsers.add_parser("caption", help="Caption a single image")
    caption_parser.add_argument("image", help="Path to image file")
    caption_parser.add_argument("--model", default="Salesforce/blip-image-captioning-base", help="Model to use")
    caption_parser.add_argument("--device", help="Device to use (cpu/cuda)")
    caption_parser.add_argument("--use-pipeline", action="store_true", default=True, help="Use Hugging Face pipeline")
    caption_parser.add_argument("--max-length", type=int, default=50, help="Maximum caption length")
    caption_parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for generation")
    caption_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    caption_parser.add_argument("--do-sample", action="store_true", help="Use sampling")
    caption_parser.add_argument("--output", help="Output file for results")
    
    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process multiple images")
    batch_parser.add_argument("input_dir", help="Directory containing images")
    batch_parser.add_argument("--model", default="Salesforce/blip-image-captioning-base", help="Model to use")
    batch_parser.add_argument("--device", help="Device to use (cpu/cuda)")
    batch_parser.add_argument("--use-pipeline", action="store_true", default=True, help="Use Hugging Face pipeline")
    batch_parser.add_argument("--max-length", type=int, default=50, help="Maximum caption length")
    batch_parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for generation")
    batch_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    batch_parser.add_argument("--do-sample", action="store_true", help="Use sampling")
    batch_parser.add_argument("--output", help="Output file for results")
    
    # Sample generation
    generate_parser = subparsers.add_parser("generate-samples", help="Generate sample images")
    generate_parser.add_argument("--num-images", type=int, default=20, help="Number of images to generate")
    generate_parser.add_argument("--categories", nargs="+", default=["animals", "objects"], help="Categories to generate")
    generate_parser.add_argument("--scenes", action="store_true", help="Generate scene images instead")
    generate_parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    
    # Advanced captioning
    advanced_parser = subparsers.add_parser("advanced", help="Advanced captioning methods")
    advanced_parser.add_argument("image", help="Path to image file")
    advanced_parser.add_argument("--method", choices=["ensemble", "zero_shot", "compare"], default="ensemble", help="Captioning method")
    advanced_parser.add_argument("--models", nargs="+", default=["Salesforce/blip-image-captioning-base"], help="Models to use")
    advanced_parser.add_argument("--device", help="Device to use (cpu/cuda)")
    advanced_parser.add_argument("--enable-ensemble", action="store_true", default=True, help="Enable ensemble methods")
    advanced_parser.add_argument("--enable-clip-reranking", action="store_true", default=True, help="Enable CLIP reranking")
    advanced_parser.add_argument("--num-captions", type=int, default=3, help="Number of captions per model")
    advanced_parser.add_argument("--strategy", choices=["voting", "weighted", "best"], default="voting", help="Ensemble strategy")
    advanced_parser.add_argument("--methods", nargs="+", default=["ensemble", "zero_shot"], help="Methods to compare")
    advanced_parser.add_argument("--output", help="Output file for results")
    
    # Web interface
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="localhost", help="Host to bind to")
    web_parser.add_argument("--port", type=int, default=8501, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == "caption":
        cmd_single_image(args)
    elif args.command == "batch":
        cmd_batch_process(args)
    elif args.command == "generate-samples":
        cmd_generate_samples(args)
    elif args.command == "advanced":
        cmd_advanced_captioning(args)
    elif args.command == "web":
        cmd_web_interface(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
