"""
Visualization utilities for image captioning results.

This module provides tools for visualizing and analyzing
image captioning results and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CaptionVisualizer:
    """Visualization tools for image captioning results."""
    
    def __init__(self, style: str = "whitegrid"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_caption_comparison(
        self, 
        image_path: str,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of different captioning methods.
        
        Args:
            image_path: Path to the image
            results: Dictionary of method results
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Load and display image
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title("Input Image", fontsize=14)
        ax1.axis('off')
        
        # Plot caption results
        methods = list(results.keys())
        captions = [results[method]['caption'] for method in methods]
        confidences = [results[method].get('confidence', 0) for method in methods]
        
        # Create caption comparison
        y_pos = np.arange(len(methods))
        ax2.barh(y_pos, confidences, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(methods)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Caption Confidence Comparison')
        
        # Add caption text
        for i, (method, caption) in enumerate(zip(methods, captions)):
            ax2.text(0.5, i, f"{caption[:50]}...", 
                    verticalalignment='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_batch_results(
        self, 
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot batch processing results.
        
        Args:
            results: List of batch processing results
            save_path: Path to save the plot
        """
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate
        success_rate = (df['status'] == 'success').mean()
        axes[0, 0].pie([success_rate, 1-success_rate], 
                      labels=['Success', 'Failed'],
                      autopct='%1.1f%%',
                      colors=['green', 'red'])
        axes[0, 0].set_title('Processing Success Rate')
        
        # Caption length distribution
        successful_df = df[df['status'] == 'success']
        if not successful_df.empty:
            caption_lengths = successful_df['caption'].str.len()
            axes[0, 1].hist(caption_lengths, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Caption Length (characters)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Caption Length Distribution')
        
        # Processing time (if available)
        if 'processing_time' in df.columns:
            axes[1, 0].plot(range(len(df)), df['processing_time'], 'o-')
            axes[1, 0].set_xlabel('Image Index')
            axes[1, 0].set_ylabel('Processing Time (seconds)')
            axes[1, 0].set_title('Processing Time per Image')
        
        # Error analysis
        error_df = df[df['status'] == 'error']
        if not error_df.empty and 'error' in error_df.columns:
            error_counts = error_df['error'].value_counts()
            axes[1, 1].bar(range(len(error_counts)), error_counts.values)
            axes[1, 1].set_xticks(range(len(error_counts)))
            axes[1, 1].set_xticklabels(error_counts.index, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Error Count')
            axes[1, 1].set_title('Error Types')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Batch results plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_performance(
        self, 
        performance_data: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot model performance comparison.
        
        Args:
            performance_data: Dictionary mapping model names to performance metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(performance_data.keys())
        
        # Processing time comparison
        if 'processing_time' in performance_data:
            times = performance_data['processing_time']
            axes[0, 0].bar(models, times, alpha=0.7)
            axes[0, 0].set_ylabel('Processing Time (seconds)')
            axes[0, 0].set_title('Processing Time Comparison')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence scores
        if 'confidence' in performance_data:
            confidences = performance_data['confidence']
            axes[0, 1].bar(models, confidences, alpha=0.7)
            axes[0, 1].set_ylabel('Average Confidence')
            axes[0, 1].set_title('Confidence Score Comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Caption quality metrics (if available)
        if 'bleu_score' in performance_data:
            bleu_scores = performance_data['bleu_score']
            axes[1, 0].bar(models, bleu_scores, alpha=0.7)
            axes[1, 0].set_ylabel('BLEU Score')
            axes[1, 0].set_title('BLEU Score Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall performance radar chart
        if len(performance_data) > 1:
            metrics = ['processing_time', 'confidence', 'bleu_score']
            available_metrics = [m for m in metrics if m in performance_data]
            
            if available_metrics:
                # Normalize metrics for radar chart
                normalized_data = {}
                for model in models:
                    normalized_data[model] = []
                    for metric in available_metrics:
                        values = performance_data[metric]
                        max_val = max(values)
                        min_val = min(values)
                        if max_val > min_val:
                            normalized_val = (values[models.index(model)] - min_val) / (max_val - min_val)
                        else:
                            normalized_val = 1.0
                        normalized_data[model].append(normalized_val)
                
                # Create radar chart
                angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
                angles += angles[:1]  # Complete the circle
                
                ax = axes[1, 1]
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_thetagrids(np.degrees(angles[:-1]), available_metrics)
                
                for model in models:
                    values = normalized_data[model] + normalized_data[model][:1]
                    ax.plot(angles, values, 'o-', linewidth=2, label=model)
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_title('Model Performance Radar Chart')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def create_caption_gallery(
        self, 
        image_caption_pairs: List[tuple],
        cols: int = 3,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a gallery of images with their captions.
        
        Args:
            image_caption_pairs: List of (image_path, caption) tuples
            cols: Number of columns in the gallery
            save_path: Path to save the gallery
        """
        n_images = len(image_caption_pairs)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (image_path, caption) in enumerate(image_caption_pairs):
            if i >= len(axes):
                break
            
            try:
                image = Image.open(image_path)
                axes[i].imshow(image)
                axes[i].set_title(caption, fontsize=10, wrap=True)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading {image_path}: {e}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Caption gallery saved to {save_path}")
        
        plt.show()


def analyze_caption_quality(captions: List[str]) -> Dict[str, float]:
    """
    Analyze the quality of generated captions.
    
    Args:
        captions: List of generated captions
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Length statistics
    lengths = [len(caption.split()) for caption in captions]
    metrics['avg_length'] = np.mean(lengths)
    metrics['length_std'] = np.std(lengths)
    metrics['min_length'] = min(lengths)
    metrics['max_length'] = max(lengths)
    
    # Diversity metrics
    all_words = []
    for caption in captions:
        words = caption.lower().split()
        all_words.extend(words)
    
    unique_words = set(all_words)
    metrics['vocabulary_size'] = len(unique_words)
    metrics['type_token_ratio'] = len(unique_words) / len(all_words) if all_words else 0
    
    # Repetition analysis
    repeated_words = 0
    for word in unique_words:
        if all_words.count(word) > 1:
            repeated_words += 1
    
    metrics['repetition_rate'] = repeated_words / len(unique_words) if unique_words else 0
    
    return metrics


def main():
    """Example usage of visualization tools."""
    visualizer = CaptionVisualizer()
    
    # Example: Create a simple gallery
    sample_dir = Path("data/synthetic")
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*.jpg"))[:6]
        
        if image_files:
            # Mock captions for demonstration
            captions = [f"Sample caption for {img.name}" for img in image_files]
            pairs = list(zip(image_files, captions))
            
            visualizer.create_caption_gallery(pairs, cols=3)
        else:
            print("No sample images found. Generate some images first.")
    else:
        print("Sample directory not found. Run the data generator first.")


if __name__ == "__main__":
    main()
