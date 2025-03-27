"""
YOLO training pipeline for PCB defect detection.
"""
import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import logging
from src.models.augmentation import PCBAugmentation

class DefectDetectorTrainer:
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.augmentation = PCBAugmentation()
        
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
    
    def train(self, dataset_config: str):
        """Train the YOLO model using prepared dataset"""
        try:
            # Initialize YOLO model
            model = YOLO(self.config['model']['name'])
            
            # Training arguments
            args = {
                'data': dataset_config,
                'epochs': self.config['training']['epochs'],
                'batch': self.config['training']['batch_size'],
                'imgsz': self.config['model']['input_size'],
                'device': self.config['model']['device'],
                'workers': self.config['model']['num_workers'],
                'patience': self.config['training']['early_stopping_patience'],
                'save': True,
                'save_period': 10,
                'cache': False,
                'project': self.config['training']['output_dir'],
                'name': self.config['training']['experiment_name'],
                'exist_ok': True,
                'pretrained': True,
                'optimizer': self.config['training']['optimizer'],
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': True,
                'close_mosaic': 0,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'dropout': self.config['training']['dropout'],
                'val': True,
                'split': 'val',
                'save_json': False,
                'save_hybrid': False,
                'conf': self.config['detection']['conf'],
                'iou': self.config['detection']['iou'],
                'max_det': self.config['detection']['max_det'],
                'plots': True,
                'lr0': self.config['training']['learning_rate'],
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': self.config['training']['weight_decay'],
                'warmup_epochs': 5.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': self.config['loss']['box'],
                'cls': self.config['loss']['cls'],
                'dfl': self.config['loss']['dfl'],
                **self.config['augmentation']
            }
            
            # Start training
            self.logger.info("Starting training...")
            results = model.train(**args)
            
            # Save training results
            self.logger.info(f"Training completed. Results saved to {self.config['training']['output_dir']}")
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