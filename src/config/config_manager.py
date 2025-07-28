"""
Configuration Manager for Detect-Track System

Handles loading, validation, and access to configuration parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['video', 'detection', 'tracking', 'logging', 'performance', 'display']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate video source
        video_source = self.config['video']['source']
        if video_source not in ['camera', 'file']:
            raise ValueError(f"Invalid video source: {video_source}. Must be 'camera' or 'file'")
        
        # Validate detection model
        detection_model = self.config['detection']['model']
        valid_models = ['yolov5n', 'yolov5s', 'yolov7', 'yolov8n', 'yolov8s']
        if detection_model not in valid_models:
            raise ValueError(f"Invalid detection model: {detection_model}. Must be one of {valid_models}")
        
        # Validate tracking algorithm
        tracking_algo = self.config['tracking']['algorithm']
        valid_trackers = ['nvdcf', 'deepsort', 'botsort']
        if tracking_algo not in valid_trackers:
            raise ValueError(f"Invalid tracking algorithm: {tracking_algo}. Must be one of {valid_trackers}")
        
        logger.info("Configuration validation successful")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration key (e.g., 'video.camera.device_id')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
        logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        save_path = output_path or self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @property
    def video_config(self) -> Dict[str, Any]:
        """Get video configuration section."""
        return self.config['video']
    
    @property
    def detection_config(self) -> Dict[str, Any]:
        """Get detection configuration section."""
        return self.config['detection']
    
    @property
    def tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration section."""
        return self.config['tracking']
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config['logging']
    
    @property
    def performance_config(self) -> Dict[str, Any]:
        """Get performance configuration section."""
        return self.config['performance']
    
    @property
    def display_config(self) -> Dict[str, Any]:
        """Get display configuration section."""
        return self.config['display'] 