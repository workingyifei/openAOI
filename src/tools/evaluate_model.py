"""
Evaluate YOLO model performance on PCB defect detection.
"""
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd
import yaml

class ModelEvaluator:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.logger = self._setup_logger()
        self.class_names = ['Bad_podu', 'Bad_qiaojiao']  # Solder paste slope and bridging defects
        
    def _setup_logger(self):
        logger = logging.getLogger('Evaluation')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger
    
    def evaluate(self, dataset_yaml: str, output_path: str = 'evaluation'):
        """Evaluate model performance on test dataset"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Load dataset configuration
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get test image paths
        test_path = Path(dataset_config['test'])
        image_files = sorted(test_path.glob('*.jpg'))
        
        # Initialize metrics
        metrics = {
            'total_images': len(image_files),
            'total_detections': 0,
            'class_metrics': {cls: {
                'tp': 0, 'fp': 0, 'fn': 0,
                'precisions': [], 'recalls': [],
                'confidences': [],
                'ap': 0.0
            } for cls in self.class_names},
            'inference_times': []
        }
        
        # Process each test image
        all_predictions = []
        all_ground_truths = []
        
        for img_file in image_files:
            self.logger.info(f"Processing {img_file}")
            
            # Load image and ground truth
            img = cv2.imread(str(img_file))
            label_file = test_path.parent / 'labels' / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                self.logger.warning(f"Ground truth not found for {img_file}")
                continue
            
            # Run inference
            results = self.model(img)[0]
            metrics['inference_times'].append(results.speed['inference'])
            
            # Get predictions and ground truths
            predictions = self._get_predictions(results)
            ground_truths = self._load_ground_truth(label_file)
            
            # Process detections
            self._process_detections(predictions, ground_truths, metrics)
            
            # Save visualization
            self._save_visualization(img, results, output_path / f"{img_file.stem}_eval.jpg")
            
            # Store for confusion matrix
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
        
        # Calculate final metrics
        final_metrics = self._calculate_metrics(metrics)
        
        # Plot metrics
        self._plot_metrics(metrics, all_predictions, all_ground_truths, output_path)
        
        # Save metrics
        self._save_metrics(final_metrics, output_path)
        
        return final_metrics
    
    def _get_predictions(self, results) -> List[Dict]:
        """Convert YOLO results to prediction format"""
        predictions = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            pred = {
                'class_id': int(box.cls),
                'confidence': float(box.conf),
                'box': box.xywh[0].tolist()  # x_center, y_center, width, height
            }
            if pred['confidence'] >= self.conf_threshold:
                predictions.append(pred)
        
        return predictions
    
    def _load_ground_truth(self, label_file: Path) -> List[Dict]:
        """Load ground truth annotations"""
        ground_truths = []
        with open(label_file, 'r') as f:
            for line in f:
                cls_id, x, y, w, h = map(float, line.strip().split())
                ground_truths.append({
                    'class_id': int(cls_id),
                    'box': [x, y, w, h]
                })
        return ground_truths
    
    def _process_detections(self, predictions: List[Dict], ground_truths: List[Dict], metrics: Dict):
        """Process detections and update metrics"""
        # Track matched ground truths to avoid double-counting
        matched_gt = set()
        
        # Process each prediction
        for pred in predictions:
            metrics['total_detections'] += 1
            cls_name = self.class_names[pred['class_id']]
            metrics['class_metrics'][cls_name]['confidences'].append(pred['confidence'])
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(ground_truths):
                if i in matched_gt or gt['class_id'] != pred['class_id']:
                    continue
                
                iou = self._calculate_iou(pred['box'], gt['box'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_gt_idx >= 0:
                # True positive
                metrics['class_metrics'][cls_name]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False positive
                metrics['class_metrics'][cls_name]['fp'] += 1
        
        # Count false negatives
        for i, gt in enumerate(ground_truths):
            if i not in matched_gt:
                cls_name = self.class_names[gt['class_id']]
                metrics['class_metrics'][cls_name]['fn'] += 1
    
    def _calculate_iou(self, box1, box2) -> float:
        """Calculate IoU between two boxes in YOLO format (x_center, y_center, width, height)"""
        # Convert to x1,y1,x2,y2 format
        b1_x1 = box1[0] - box1[2]/2
        b1_y1 = box1[1] - box1[3]/2
        b1_x2 = box1[0] + box1[2]/2
        b1_y2 = box1[1] + box1[3]/2
        
        b2_x1 = box2[0] - box2[2]/2
        b2_y1 = box2[1] - box2[3]/2
        b2_x2 = box2[0] + box2[2]/2
        b2_y2 = box2[1] + box2[3]/2
        
        # Calculate intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        
        return inter_area / (b1_area + b2_area - inter_area)
    
    def _calculate_metrics(self, metrics: Dict) -> Dict:
        """Calculate final metrics"""
        final_metrics = {
            'total_images': metrics['total_images'],
            'total_detections': metrics['total_detections'],
            'avg_inference_time': np.mean(metrics['inference_times']),
            'classes': {}
        }
        
        for cls_name, cls_metrics in metrics['class_metrics'].items():
            tp = cls_metrics['tp']
            fp = cls_metrics['fp']
            fn = cls_metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AP using sklearn
            if cls_metrics['confidences']:
                y_true = [1] * tp + [0] * fp
                y_scores = sorted(cls_metrics['confidences'], reverse=True)
                ap = average_precision_score(y_true, y_scores)
            else:
                ap = 0.0
            
            final_metrics['classes'][cls_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'ap': ap,
                'total_predictions': tp + fp,
                'total_ground_truths': tp + fn
            }
        
        # Calculate mAP
        final_metrics['mAP'] = np.mean([
            cls_metrics['ap'] for cls_metrics in final_metrics['classes'].values()
        ])
        
        return final_metrics
    
    def _plot_metrics(self, metrics: Dict, predictions: List[Dict], ground_truths: List[Dict], output_path: Path):
        """Plot evaluation metrics"""
        # 1. Confusion Matrix
        y_true = [gt['class_id'] for gt in ground_truths]
        y_pred = [pred['class_id'] for pred in predictions]
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_path / 'confusion_matrix.png')
        plt.close()
        
        # 2. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for cls_name, cls_metrics in metrics['class_metrics'].items():
            if cls_metrics['confidences']:
                y_true = [1] * cls_metrics['tp'] + [0] * cls_metrics['fp']
                y_scores = sorted(cls_metrics['confidences'], reverse=True)
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                plt.plot(recall, precision, label=f'{cls_name} (AP={cls_metrics["ap"]:.3f})')
        
        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / 'precision_recall_curves.png')
        plt.close()
        
        # 3. Confidence Distribution
        plt.figure(figsize=(10, 8))
        for cls_name, cls_metrics in metrics['class_metrics'].items():
            if cls_metrics['confidences']:
                plt.hist(cls_metrics['confidences'], bins=20, alpha=0.5, label=cls_name)
        
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / 'confidence_distribution.png')
        plt.close()
    
    def _save_metrics(self, metrics: Dict, output_path: Path):
        """Save metrics to JSON file"""
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_visualization(self, img: np.ndarray, results, output_path: Path):
        """Save detection visualization"""
        # Get the plotted image from YOLO results
        vis_img = results.plot()
        cv2.imwrite(str(output_path), vis_img)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate PCB defect detection model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained YOLO model')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to dataset YAML file')
    parser.add_argument('--output', type=str, default='evaluation',
                      help='Output directory for evaluation results')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                      help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                      help='IoU threshold for matching predictions with ground truth')
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.conf_thres, args.iou_thres)
    metrics = evaluator.evaluate(args.dataset, args.output)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total Images: {metrics['total_images']}")
    print(f"Total Detections: {metrics['total_detections']}")
    print(f"Average Inference Time: {metrics['avg_inference_time']:.2f}ms")
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.3f}")
    
    print("\nPer-Class Metrics:")
    for cls_name, cls_metrics in metrics['classes'].items():
        print(f"\n{cls_name}:")
        print(f"  Precision: {cls_metrics['precision']:.3f}")
        print(f"  Recall: {cls_metrics['recall']:.3f}")
        print(f"  F1-Score: {cls_metrics['f1_score']:.3f}")
        print(f"  Average Precision: {cls_metrics['ap']:.3f}")
        print(f"  Total Predictions: {cls_metrics['total_predictions']}")
        print(f"  Total Ground Truths: {cls_metrics['total_ground_truths']}")

if __name__ == '__main__':
    main() 