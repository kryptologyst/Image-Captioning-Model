"""
Configuration module for the Image Captioning project.

This module handles all configuration settings using YAML files
and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the image captioning model."""
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: Optional[str] = None
    use_pipeline: bool = True
    max_length: int = 50
    num_beams: int = 4
    temperature: float = 1.0
    do_sample: bool = False


@dataclass
class AppConfig:
    """Configuration for the application."""
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "output"


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.model_config = ModelConfig()
        self.app_config = AppConfig()
        
        self._load_config()
        self._load_env_vars()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    self._update_from_dict(config_data)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        # Model configuration
        if os.getenv("MODEL_NAME"):
            self.model_config.model_name = os.getenv("MODEL_NAME")
        
        if os.getenv("DEVICE"):
            self.model_config.device = os.getenv("DEVICE")
        
        if os.getenv("USE_PIPELINE"):
            self.model_config.use_pipeline = os.getenv("USE_PIPELINE").lower() == "true"
        
        # App configuration
        if os.getenv("DEBUG"):
            self.app_config.debug = os.getenv("DEBUG").lower() == "true"
        
        if os.getenv("LOG_LEVEL"):
            self.app_config.log_level = os.getenv("LOG_LEVEL")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        # Update model config
        if "model" in config_data:
            model_data = config_data["model"]
            for key, value in model_data.items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        # Update app config
        if "app" in config_data:
            app_data = config_data["app"]
            for key, value in app_data.items():
                if hasattr(self.app_config, key):
                    setattr(self.app_config, key, value)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to YAML file."""
        save_path = path or self.config_path
        
        config_data = {
            "model": {
                "model_name": self.model_config.model_name,
                "device": self.model_config.device,
                "use_pipeline": self.model_config.use_pipeline,
                "max_length": self.model_config.max_length,
                "num_beams": self.model_config.num_beams,
                "temperature": self.model_config.temperature,
                "do_sample": self.model_config.do_sample,
            },
            "app": {
                "debug": self.app_config.debug,
                "log_level": self.app_config.log_level,
                "data_dir": self.app_config.data_dir,
                "models_dir": self.app_config.models_dir,
                "output_dir": self.app_config.output_dir,
            }
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model configuration as keyword arguments."""
        return {
            "model_name": self.model_config.model_name,
            "device": self.model_config.device,
            "use_pipeline": self.model_config.use_pipeline,
        }
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation configuration as keyword arguments."""
        return {
            "max_length": self.model_config.max_length,
            "num_beams": self.model_config.num_beams,
            "temperature": self.model_config.temperature,
            "do_sample": self.model_config.do_sample,
        }


# Global configuration instance
config = Config()
