"""
Configuration settings for the OpenAOI system.
Handles all settings including camera, processing, and visualization parameters.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import cv2

class ValidationError(Exception):
    """Raised when settings validation fails"""
    pass

class Settings:
    def __init__(self, config_path: Optional[str] = None):
        # Default paths
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 
            '..',
            'qa',
            'ipc610_config.py'
        )
        
        # Camera settings
        self.CAMERA_INDEX: int = 0
        self.EXPOSURE_TIME: float = 100.0  # ms
        self.GAIN: float = 1.0
        
        # Processing settings
        self.PIXEL_TO_MM_RATIO: float = 0.1  # mm per pixel
        self.MIN_CONFIDENCE: float = 0.5
        self.NMS_THRESHOLD: float = 0.3
        self.INPUT_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
        self.USE_CAMERA: bool = False
        self.TEST_IMAGE_PATH: str = "test_data/sample_pcb.jpg"
        self.TEMPLATE_PATH: str = "test_data/pcb_template.jpg"
        
        # Visualization settings
        self.DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
            'SolderBridging': (255, 0, 0),
            'InsufficientSolder': (0, 255, 0),
            'ExcessSolder': (0, 0, 255),
            'Voiding': (255, 255, 0),
            'Tombstoning': (255, 0, 255),
            'ComponentShift': (0, 255, 255),
            'ComponentRotation': (128, 0, 0),
            'Contamination': (0, 128, 0),
            'SolderBall': (0, 0, 128),
            'PadLift': (128, 128, 0),
            'CopperExposure': (128, 0, 128),
            'LeadBend': (0, 128, 128),
            'LeadProtrusion': (128, 128, 128)
        }
        self.SAVE_DEBUG_IMAGES: bool = True
        self.OUTPUT_DIR: str = "results"
        self.INSPECTION_ANGLES = [0, 45, 90, 135]  # degrees
    
    def validate(self) -> None:
        """Validate all settings"""
        self._validate_camera_settings()
        self._validate_processing_settings()
        self._validate_visualization_settings()
        self._validate_paths()
    
    def _validate_camera_settings(self) -> None:
        """Validate camera-related settings"""
        if not isinstance(self.CAMERA_INDEX, int) or self.CAMERA_INDEX < 0:
            raise ValidationError("Camera index must be a non-negative integer")
        
        if not isinstance(self.EXPOSURE_TIME, (int, float)) or self.EXPOSURE_TIME <= 0:
            raise ValidationError("Exposure time must be a positive number")
        
        if not isinstance(self.GAIN, (int, float)) or self.GAIN <= 0:
            raise ValidationError("Gain must be a positive number")
    
    def _validate_processing_settings(self) -> None:
        """Validate processing-related settings"""
        if not isinstance(self.PIXEL_TO_MM_RATIO, (int, float)) or self.PIXEL_TO_MM_RATIO <= 0:
            raise ValidationError("Pixel to mm ratio must be a positive number")
        
        if not 0 <= self.MIN_CONFIDENCE <= 1:
            raise ValidationError("Minimum confidence must be between 0 and 1")
        
        if not 0 <= self.NMS_THRESHOLD <= 1:
            raise ValidationError("NMS threshold must be between 0 and 1")
        
        if not isinstance(self.INPUT_IMAGE_SIZE, (tuple, list)) or len(self.INPUT_IMAGE_SIZE) != 2:
            raise ValidationError("Input image size must be a tuple of two integers")
    
    def _validate_visualization_settings(self) -> None:
        """Validate visualization-related settings"""
        for defect_type, color in self.DEFECT_COLORS.items():
            if not isinstance(color, (tuple, list)) or len(color) != 3:
                raise ValidationError(f"Invalid color format for defect type {defect_type}")
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                raise ValidationError(f"Color values must be integers between 0 and 255")
    
    def _validate_paths(self) -> None:
        """Validate file and directory paths"""
        if not self.USE_CAMERA:
            if not os.path.exists(self.TEST_IMAGE_PATH):
                raise ValidationError(f"Test image not found: {self.TEST_IMAGE_PATH}")
            if not os.path.exists(self.TEMPLATE_PATH):
                raise ValidationError(f"Template image not found: {self.TEMPLATE_PATH}")
    
    def load_ipc_classes(self) -> Dict:
        """Load IPC-610 defect class definitions"""
        try:
            config_path = Path(self.config_path)
            namespace = {}
            with open(config_path, 'r') as f:
                exec(f.read(), namespace)
            
            return namespace.get('IPC610_DEFECT_CLASSES', {})
        except Exception as e:
            raise RuntimeError(f"Failed to load IPC classes: {str(e)}")
    
    def save_settings(self, filepath: str) -> None:
        """Save current settings to a JSON file"""
        settings_dict = {
            'camera': {
                'index': self.CAMERA_INDEX,
                'exposure_time': self.EXPOSURE_TIME,
                'gain': self.GAIN
            },
            'processing': {
                'pixel_to_mm_ratio': self.PIXEL_TO_MM_RATIO,
                'min_confidence': self.MIN_CONFIDENCE,
                'nms_threshold': self.NMS_THRESHOLD,
                'input_image_size': list(self.INPUT_IMAGE_SIZE),
                'use_camera': self.USE_CAMERA,
                'test_image_path': self.TEST_IMAGE_PATH,
                'template_path': self.TEMPLATE_PATH
            },
            'visualization': {
                'defect_colors': {k: list(v) for k, v in self.DEFECT_COLORS.items()},
                'save_debug_images': self.SAVE_DEBUG_IMAGES,
                'output_dir': self.OUTPUT_DIR
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(settings_dict, f, indent=4)
    
    @classmethod
    def load_settings(cls, filepath: str) -> 'Settings':
        """Load settings from a JSON file"""
        with open(filepath, 'r') as f:
            settings_dict = json.load(f)
        
        settings = cls()
        
        # Camera settings
        camera = settings_dict.get('camera', {})
        settings.CAMERA_INDEX = camera.get('index', settings.CAMERA_INDEX)
        settings.EXPOSURE_TIME = camera.get('exposure_time', settings.EXPOSURE_TIME)
        settings.GAIN = camera.get('gain', settings.GAIN)
        
        # Processing settings
        processing = settings_dict.get('processing', {})
        settings.PIXEL_TO_MM_RATIO = processing.get('pixel_to_mm_ratio', settings.PIXEL_TO_MM_RATIO)
        settings.MIN_CONFIDENCE = processing.get('min_confidence', settings.MIN_CONFIDENCE)
        settings.NMS_THRESHOLD = processing.get('nms_threshold', settings.NMS_THRESHOLD)
        settings.INPUT_IMAGE_SIZE = tuple(processing.get('input_image_size', settings.INPUT_IMAGE_SIZE))
        settings.USE_CAMERA = processing.get('use_camera', settings.USE_CAMERA)
        settings.TEST_IMAGE_PATH = processing.get('test_image_path', settings.TEST_IMAGE_PATH)
        settings.TEMPLATE_PATH = processing.get('template_path', settings.TEMPLATE_PATH)
        
        # Visualization settings
        viz = settings_dict.get('visualization', {})
        settings.DEFECT_COLORS = {k: tuple(v) for k, v in viz.get('defect_colors', settings.DEFECT_COLORS).items()}
        settings.SAVE_DEBUG_IMAGES = viz.get('save_debug_images', settings.SAVE_DEBUG_IMAGES)
        settings.OUTPUT_DIR = viz.get('output_dir', settings.OUTPUT_DIR)
        
        # Validate all settings
        settings.validate()
        
        return settings 