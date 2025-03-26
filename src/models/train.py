"""
YOLO training pipeline for PCB defect detection.
"""
import os
import yaml
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import albumentations as A
import logging

class PCBAugmentation:
    """PCB-specific data augmentation pipeline"""
    def __init__(self, config=None):
        self.config = config or {}
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create augmentation pipeline"""
        return A.Compose([
            # Geometric transforms
            A.Affine(
                rotate=30,
                scale=(0.8, 1.2),
                translate_percent=(0.1, 0.1),
                p=0.5
            ),
            
            # Color/intensity transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=0,  # No hue shift for PCB images
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            
            # PCB-specific transforms
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomShadow(p=0.2),
        ])
    
    def __call__(self, image):
        """Apply augmentation to image"""
        return self.transform(image=image)['image']

class DefectDetectorTrainer:
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.augmentation = PCBAugmentation(self.config.get('augmentation'))
        
    def _setup_logger(self):
        logger = logging.getLogger('Training')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger
    
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_dataset(self, data_dir: str, split_ratio: float = 0.2):
        """
        Prepare dataset for training.
        Args:
            data_dir: Root directory containing images and annotations
            split_ratio: Ratio for validation split
        """
        # Get all image paths
        image_paths = []
        for defect_type in os.listdir(os.path.join(data_dir, 'samples/defects')):
            defect_dir = os.path.join(data_dir, 'samples/defects', defect_type)
            if os.path.isdir(defect_dir):
                image_paths.extend(
                    [str(p) for p in Path(defect_dir).glob('*.jpg')]
                )
        
        # Add good samples
        good_dir = os.path.join(data_dir, 'samples/good')
        image_paths.extend([str(p) for p in Path(good_dir).glob('*.jpg')])
        
        # Split dataset
        train_paths, val_paths = train_test_split(
            image_paths, 
            test_size=split_ratio,
            random_state=42
        )
        
        # Create YOLO dataset config
        dataset_config = {
            'path': data_dir,
            'train': train_paths,
            'val': val_paths,
            'names': self.config['defect_classes']
        }
        
        # Save dataset config
        with open('config/dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
            
        self.logger.info(f"Dataset prepared: {len(train_paths)} train, {len(val_paths)} validation images")
        return dataset_config
    
    def train(self, dataset_config: str):
        """Train the YOLO model"""
        try:
            # Initialize YOLO model
            model = YOLO('yolov8s.pt')
            
            # Training arguments
            args = {
                'data': dataset_config,
                'epochs': self.config['epochs'],
                'batch': self.config['batch_size'],
                'imgsz': self.config['image_size'],
                'device': self.config['device'],
                'workers': self.config['num_workers'],
                'patience': self.config['early_stopping_patience'],
                'save': True,
                'save_period': 10,  # Save every 10 epochs
                'cache': False,
                'project': self.config['output_dir'],
                'name': self.config['experiment_name'],
                'exist_ok': True,
                'pretrained': True,
                'optimizer': self.config['optimizer'],
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': False,
                'rect': False,  # No rectangular training
                'cos_lr': True,  # Cosine LR scheduler
                'close_mosaic': 0,  # Disable mosaic augmentation
                'resume': False,
                'amp': True,  # Mixed precision training
                'fraction': 1.0,
                'dropout': self.config['dropout'],
                'val': True,
                'split': 'val',
                'save_json': False,
                'save_hybrid': False,
                'conf': 0.001,  # Lower confidence for training
                'iou': 0.7,
                'max_det': 300,
                'plots': True,
                'lr0': self.config['learning_rate'],
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': self.config['weight_decay'],
                'warmup_epochs': 5.0,  # Longer warmup
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'hsv_h': 0.015,  # Reduced color augmentation
                'hsv_s': 0.4,
                'hsv_v': 0.4,
                'degrees': 0.0,  # No rotation
                'translate': 0.1,
                'scale': 0.3,  # Reduced scale augmentation
                'shear': 0.0,  # No shear
                'perspective': 0.0,  # No perspective
                'flipud': 0.0,  # No vertical flip
                'fliplr': 0.5,  # Keep horizontal flip
                'mosaic': 0.0,  # Disable mosaic
                'mixup': 0.0,  # Disable mixup
                'copy_paste': 0.0  # Disable copy-paste
            }
            
            # Start training
            self.logger.info("Starting training...")
            results = model.train(**args)
            
            # Save training results
            self.logger.info(f"Training completed. Results saved to {self.config['output_dir']}")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train PCB defect detection model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to model configuration file')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to dataset configuration file')
    args = parser.parse_args()
    
    trainer = DefectDetectorTrainer(args.config)
    results = trainer.train(args.dataset)

if __name__ == '__main__':
    main() 