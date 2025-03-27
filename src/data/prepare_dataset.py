"""
Convert XML annotations to YOLO format and prepare dataset for training.
"""
import os
import xml.etree.ElementTree as ET
import glob
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import logging
import yaml

# Define defect classes based on the dataset
DEFECT_CLASSES = {
    'SolderSlopeDefect': 0,      # Solder paste slope defect (坡度)
    'SolderBridgingDefect': 1,   # Solder bridging defect (桥脚)
    # Legacy mappings
    'Bad_podu': 0,               # Maps to SolderSlopeDefect
    'Bad_qiaojiao': 1           # Maps to SolderBridgingDefect
}

class DatasetPreparator:
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path).absolute()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('DataPreparation')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger
    
    def prepare(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Prepare the dataset for training, validation, and testing.
        
        Args:
            train_ratio: Ratio of training set size (default: 0.7)
            val_ratio: Ratio of validation set size (default: 0.15)
            The remaining (1 - train_ratio - val_ratio) will be used for testing
        """
        # Create output directories
        self._create_directories()
        
        # Get all image files
        image_files = sorted(self.dataset_path.glob('train_data/JPEGImages/*.jpeg'))
        
        # First split into train and temp (val + test)
        train_files, temp_files = train_test_split(
            image_files,
            train_size=train_ratio,
            random_state=42
        )
        
        # Split temp into val and test
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_ratio_adjusted,
            random_state=42
        )
        
        # Process each split
        self._process_file_set(train_files, 'train')
        self._process_file_set(val_files, 'val')
        self._process_file_set(test_files, 'test')
        
        # Create dataset configuration
        self._create_dataset_config()
        
        # Log split sizes
        self.logger.info(f"Dataset split sizes:")
        self.logger.info(f"  Train: {len(train_files)} images")
        self.logger.info(f"  Val: {len(val_files)} images")
        self.logger.info(f"  Test: {len(test_files)} images")
        self.logger.info(f"Dataset preparation completed. Files saved to {self.output_path}")
    
    def _create_directories(self):
        """Create necessary directories"""
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _process_file_set(self, image_files, split: str):
        """Process a set of files for train, validation, or test split"""
        for img_file in image_files:
            # Get corresponding annotation file
            xml_file = self.dataset_path / 'train_data/Annotations' / f"{img_file.stem}.xml"
            
            if not xml_file.exists():
                self.logger.warning(f"Annotation not found for {img_file}")
                continue
            
            # Convert annotation to YOLO format
            yolo_labels = self._convert_annotation(xml_file)
            
            if not yolo_labels:
                self.logger.warning(f"No valid annotations in {xml_file}")
                continue
            
            # Copy image
            shutil.copy2(
                img_file,
                self.output_path / split / 'images' / f"{img_file.stem}.jpg"
            )
            
            # Save YOLO labels
            label_file = self.output_path / split / 'labels' / f"{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    def _convert_annotation(self, xml_file: Path) -> list:
        """Convert XML annotation to YOLO format"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)
        
        yolo_labels = []
        
        for obj in root.findall('object'):
            # Try both 'name' and 'n' tags for defect type
            name_elem = obj.find('name')
            if name_elem is None:
                name_elem = obj.find('n')
            
            if name_elem is None:
                self.logger.warning(f"No name/n element found in object in {xml_file}")
                continue
                
            # Clean up the defect type text (remove any extra tags)
            defect_type = name_elem.text
            if defect_type:
                defect_type = defect_type.split('<')[0].strip()  # Remove any trailing tags
            else:
                self.logger.warning(f"Empty defect type in {xml_file}")
                continue
            
            if defect_type not in DEFECT_CLASSES:
                self.logger.warning(f"Unknown defect type {defect_type} in {xml_file}")
                continue
            
            class_id = DEFECT_CLASSES[defect_type]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Add YOLO label
            yolo_labels.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
        
        return yolo_labels
    
    def _create_dataset_config(self):
        """Create YAML configuration for YOLO training"""
        # Create a mapping using only the canonical names
        canonical_names = {
            0: 'SolderSlopeDefect',
            1: 'SolderBridgingDefect'
        }
        
        config = {
            'path': str(self.output_path),
            'train': str(self.output_path / 'train' / 'images'),
            'val': str(self.output_path / 'val' / 'images'),
            'test': str(self.output_path / 'test' / 'images'),
            'names': canonical_names
        }
        
        with open(self.output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare PCB defect dataset')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for prepared dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                      help='Ratio of training set size')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                      help='Ratio of validation set size')
    args = parser.parse_args()
    
    preparator = DatasetPreparator(args.dataset, args.output)
    preparator.prepare(train_ratio=args.train_ratio, val_ratio=args.val_ratio)

if __name__ == '__main__':
    main() 