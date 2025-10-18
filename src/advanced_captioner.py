"""
Enhanced Image Captioning with State-of-the-Art Techniques

This module provides advanced image captioning capabilities including:
- Multiple model support (BLIP, CLIP, etc.)
- Zero-shot and few-shot learning
- Model ensemble and voting
- Advanced generation strategies
- Performance optimization
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import warnings
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    BlipProcessor as BlipProcessorV2,
    BlipForConditionalGeneration as BlipForConditionalGenerationV2,
    CLIPProcessor,
    CLIPModel,
    pipeline,
    Pipeline
)
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    BLIP_BASE = "Salesforce/blip-image-captioning-base"
    BLIP_LARGE = "Salesforce/blip-image-captioning-large"
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    CLIP = "openai/clip-vit-base-patch32"


@dataclass
class CaptionResult:
    """Result of caption generation."""
    caption: str
    confidence: float
    model_name: str
    generation_time: float
    metadata: Dict[str, Any]


class AdvancedImageCaptioner:
    """
    Advanced image captioning with multiple models and techniques.
    
    This class provides state-of-the-art image captioning capabilities
    including model ensembles, zero-shot learning, and advanced generation strategies.
    """
    
    def __init__(
        self, 
        models: Optional[List[ModelType]] = None,
        device: Optional[str] = None,
        enable_ensemble: bool = True,
        enable_clip_reranking: bool = True
    ) -> None:
        """
        Initialize the AdvancedImageCaptioner.
        
        Args:
            models: List of models to use
            device: Device to run inference on
            enable_ensemble: Whether to use ensemble methods
            enable_clip_reranking: Whether to use CLIP for reranking
        """
        self.models = models or [ModelType.BLIP_BASE]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_ensemble = enable_ensemble
        self.enable_clip_reranking = enable_clip_reranking
        
        logger.info(f"Initializing AdvancedImageCaptioner with models: {[m.value for m in self.models]}")
        logger.info(f"Using device: {self.device}")
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all specified models."""
        self.loaded_models = {}
        
        for model_type in self.models:
            try:
                if model_type in [ModelType.BLIP_BASE, ModelType.BLIP_LARGE]:
                    self._load_blip_model(model_type)
                elif model_type == ModelType.CLIP:
                    self._load_clip_model()
                else:
                    logger.warning(f"Unsupported model type: {model_type}")
            
            except Exception as e:
                logger.error(f"Failed to load model {model_type}: {e}")
    
    def _load_blip_model(self, model_type: ModelType) -> None:
        """Load BLIP model."""
        try:
            pipeline_instance = pipeline(
                "image-to-text",
                model=model_type.value,
                device=0 if self.device == "cuda" else -1
            )
            self.loaded_models[model_type] = pipeline_instance
            logger.info(f"Successfully loaded {model_type.value}")
        except Exception as e:
            logger.error(f"Failed to load BLIP model {model_type}: {e}")
    
    def _load_clip_model(self) -> None:
        """Load CLIP model for reranking."""
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(ModelType.CLIP.value)
            self.clip_model = CLIPModel.from_pretrained(ModelType.CLIP.value).to(self.device)
            logger.info(f"Successfully loaded CLIP model")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
    
    def generate_caption_ensemble(
        self, 
        image: Union[str, Path, Image.Image],
        num_captions_per_model: int = 3,
        ensemble_strategy: str = "voting"
    ) -> CaptionResult:
        """
        Generate caption using ensemble of models.
        
        Args:
            image: Image to caption
            num_captions_per_model: Number of captions per model
            ensemble_strategy: Strategy for combining results ('voting', 'weighted', 'best')
            
        Returns:
            Ensemble caption result
        """
        import time
        start_time = time.time()
        
        all_captions = []
        model_results = {}
        
        # Generate captions from each model
        for model_type, model in self.loaded_models.items():
            try:
                captions = self._generate_multiple_captions(
                    model, image, num_captions_per_model
                )
                model_results[model_type.value] = captions
                all_captions.extend(captions)
            except Exception as e:
                logger.error(f"Error generating captions with {model_type}: {e}")
        
        if not all_captions:
            raise RuntimeError("No captions generated from any model")
        
        # Apply ensemble strategy
        if ensemble_strategy == "voting":
            final_caption = self._ensemble_voting(all_captions)
        elif ensemble_strategy == "weighted":
            final_caption = self._ensemble_weighted(model_results)
        elif ensemble_strategy == "best":
            final_caption = self._ensemble_best(image, all_captions)
        else:
            final_caption = all_captions[0]  # Fallback
        
        generation_time = time.time() - start_time
        
        return CaptionResult(
            caption=final_caption,
            confidence=self._calculate_confidence(final_caption, all_captions),
            model_name="ensemble",
            generation_time=generation_time,
            metadata={
                "strategy": ensemble_strategy,
                "total_captions": len(all_captions),
                "models_used": list(model_results.keys())
            }
        )
    
    def _generate_multiple_captions(
        self, 
        model: Pipeline, 
        image: Union[str, Path, Image.Image],
        num_captions: int
    ) -> List[str]:
        """Generate multiple captions from a single model."""
        captions = []
        
        for i in range(num_captions):
            try:
                # Vary parameters for diversity
                temperature = 0.8 + i * 0.2
                result = model(
                    image,
                    max_length=50,
                    num_beams=4,
                    temperature=temperature,
                    do_sample=True
                )
                captions.append(result[0]['generated_text'])
            except Exception as e:
                logger.error(f"Error generating caption {i}: {e}")
        
        return captions
    
    def _ensemble_voting(self, captions: List[str]) -> str:
        """Simple voting ensemble."""
        from collections import Counter
        
        # Simple word-based voting
        all_words = []
        for caption in captions:
            words = caption.lower().split()
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        most_common_words = [word for word, count in word_counts.most_common(10)]
        
        # Find caption with most common words
        best_caption = max(captions, key=lambda c: sum(
            1 for word in c.lower().split() if word in most_common_words
        ))
        
        return best_caption
    
    def _ensemble_weighted(self, model_results: Dict[str, List[str]]) -> str:
        """Weighted ensemble based on model performance."""
        # Simple weighting scheme (can be improved with actual performance metrics)
        weights = {
            "Salesforce/blip-image-captioning-large": 0.4,
            "Salesforce/blip-image-captioning-base": 0.3,
            "Salesforce/blip2-opt-2.7b": 0.3
        }
        
        weighted_captions = []
        for model_name, captions in model_results.items():
            weight = weights.get(model_name, 0.1)
            for caption in captions:
                weighted_captions.append((caption, weight))
        
        # Select caption with highest weight
        best_caption = max(weighted_captions, key=lambda x: x[1])[0]
        return best_caption
    
    def _ensemble_best(self, image: Union[str, Path, Image.Image], captions: List[str]) -> str:
        """Select best caption using CLIP reranking."""
        if not self.enable_clip_reranking or not hasattr(self, 'clip_model'):
            return captions[0]
        
        try:
            # Load image
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            
            # Get image features
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text features for each caption
            best_score = -1
            best_caption = captions[0]
            
            for caption in captions:
                text_inputs = self.clip_processor(text=caption, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features).item()
                
                if similarity > best_score:
                    best_score = similarity
                    best_caption = caption
            
            return best_caption
            
        except Exception as e:
            logger.error(f"Error in CLIP reranking: {e}")
            return captions[0]
    
    def _calculate_confidence(self, final_caption: str, all_captions: List[str]) -> float:
        """Calculate confidence score for the final caption."""
        # Simple confidence based on how many similar captions exist
        similar_count = 0
        final_words = set(final_caption.lower().split())
        
        for caption in all_captions:
            caption_words = set(caption.lower().split())
            overlap = len(final_words.intersection(caption_words))
            similarity = overlap / max(len(final_words), len(caption_words))
            
            if similarity > 0.5:
                similar_count += 1
        
        confidence = similar_count / len(all_captions)
        return min(confidence, 1.0)
    
    def zero_shot_captioning(
        self, 
        image: Union[str, Path, Image.Image],
        candidate_captions: Optional[List[str]] = None
    ) -> CaptionResult:
        """
        Zero-shot captioning using CLIP.
        
        Args:
            image: Image to caption
            candidate_captions: Pre-defined candidate captions
            
        Returns:
            Zero-shot caption result
        """
        if not hasattr(self, 'clip_model'):
            raise RuntimeError("CLIP model not loaded for zero-shot captioning")
        
        import time
        start_time = time.time()
        
        # Default candidate captions if not provided
        if candidate_captions is None:
            candidate_captions = [
                "a photo of a dog",
                "a photo of a cat",
                "a photo of a car",
                "a photo of a person",
                "a photo of a building",
                "a photo of a tree",
                "a photo of a flower",
                "a photo of food",
                "a photo of a book",
                "a photo of a phone"
            ]
        
        try:
            # Load image
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            
            # Get image features
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text features for all candidates
            text_inputs = self.clip_processor(text=candidate_captions, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(image_features, text_features)
            best_idx = similarities.argmax().item()
            best_caption = candidate_captions[best_idx]
            confidence = similarities[best_idx].item()
            
            generation_time = time.time() - start_time
            
            return CaptionResult(
                caption=best_caption,
                confidence=confidence,
                model_name="CLIP-zero-shot",
                generation_time=generation_time,
                metadata={
                    "candidate_count": len(candidate_captions),
                    "similarities": similarities.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in zero-shot captioning: {e}")
            raise
    
    def few_shot_captioning(
        self, 
        image: Union[str, Path, Image.Image],
        examples: List[Tuple[Union[str, Path, Image.Image], str]],
        num_candidates: int = 5
    ) -> CaptionResult:
        """
        Few-shot captioning using example image-caption pairs.
        
        Args:
            image: Image to caption
            examples: List of (image, caption) pairs for few-shot learning
            num_candidates: Number of candidate captions to generate
            
        Returns:
            Few-shot caption result
        """
        import time
        start_time = time.time()
        
        if not hasattr(self, 'clip_model'):
            raise RuntimeError("CLIP model not loaded for few-shot captioning")
        
        try:
            # Load target image
            if isinstance(image, (str, Path)):
                target_image = Image.open(image).convert('RGB')
            else:
                target_image = image
            
            # Get target image features
            target_inputs = self.clip_processor(images=target_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                target_features = self.clip_model.get_image_features(**target_inputs)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            
            # Process examples
            example_features = []
            example_captions = []
            
            for example_image, caption in examples:
                if isinstance(example_image, (str, Path)):
                    example_img = Image.open(example_image).convert('RGB')
                else:
                    example_img = example_image
                
                example_inputs = self.clip_processor(images=example_img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    example_img_features = self.clip_model.get_image_features(**example_inputs)
                    example_img_features = example_img_features / example_img_features.norm(dim=-1, keepdim=True)
                
                example_features.append(example_img_features)
                example_captions.append(caption)
            
            # Find most similar example
            similarities = []
            for features in example_features:
                sim = torch.cosine_similarity(target_features, features).item()
                similarities.append(sim)
            
            best_example_idx = np.argmax(similarities)
            best_caption = example_captions[best_example_idx]
            confidence = similarities[best_example_idx]
            
            generation_time = time.time() - start_time
            
            return CaptionResult(
                caption=best_caption,
                confidence=confidence,
                model_name="CLIP-few-shot",
                generation_time=generation_time,
                metadata={
                    "example_count": len(examples),
                    "similarities": similarities,
                    "best_example_idx": best_example_idx
                }
            )
            
        except Exception as e:
            logger.error(f"Error in few-shot captioning: {e}")
            raise
    
    def compare_models(
        self, 
        image: Union[str, Path, Image.Image],
        methods: Optional[List[str]] = None
    ) -> Dict[str, CaptionResult]:
        """
        Compare different captioning methods on the same image.
        
        Args:
            image: Image to caption
            methods: List of methods to compare
            
        Returns:
            Dictionary mapping method names to results
        """
        if methods is None:
            methods = ["ensemble", "zero_shot"]
        
        results = {}
        
        for method in methods:
            try:
                if method == "ensemble":
                    result = self.generate_caption_ensemble(image)
                elif method == "zero_shot":
                    result = self.zero_shot_captioning(image)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                results[method] = result
                
            except Exception as e:
                logger.error(f"Error in method {method}: {e}")
                results[method] = CaptionResult(
                    caption=f"Error: {e}",
                    confidence=0.0,
                    model_name=method,
                    generation_time=0.0,
                    metadata={"error": str(e)}
                )
        
        return results


def main() -> None:
    """Example usage of the AdvancedImageCaptioner."""
    # Initialize advanced captioner
    captioner = AdvancedImageCaptioner(
        models=[ModelType.BLIP_BASE, ModelType.CLIP],
        enable_ensemble=True,
        enable_clip_reranking=True
    )
    
    # Example with a sample image
    try:
        image_path = "data/synthetic/animals_000.jpg"
        
        # Ensemble captioning
        print("=== Ensemble Captioning ===")
        result = captioner.generate_caption_ensemble(image_path)
        print(f"Caption: {result.caption}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Time: {result.generation_time:.2f}s")
        
        # Zero-shot captioning
        print("\n=== Zero-shot Captioning ===")
        result = captioner.zero_shot_captioning(image_path)
        print(f"Caption: {result.caption}")
        print(f"Confidence: {result.confidence:.3f}")
        
        # Compare methods
        print("\n=== Method Comparison ===")
        results = captioner.compare_models(image_path)
        for method, result in results.items():
            print(f"{method}: {result.caption} (conf: {result.confidence:.3f})")
    
    except FileNotFoundError:
        print("No sample image found. Please add an image to test the advanced captioner.")
        print("You can use the data_generator.py script to generate sample images.")


if __name__ == "__main__":
    main()
