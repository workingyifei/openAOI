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
        self.class_names = ['SolderSlopeDefect', 'SolderBridgingDefect']  # Solder paste slope and bridging defects
        # Blue for ground truth, Red for predictions
        self.gt_color = (255, 128, 0)  # Blue
        self.pred_color = (0, 0, 255)  # Red
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        self.dataset_config = None  # Will store dataset configuration
        
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
            self.dataset_config = yaml.safe_load(f)
        
        # Get test image paths
        test_path = Path(self.dataset_config['path']) / self.dataset_config['test']
        image_files = sorted(test_path.glob('*.jpg'))
        
        if not image_files:
            self.logger.error(f"No images found in {test_path}")
            return None
            
        self.logger.info(f"Found {len(image_files)} test images")
        
        # Initialize metrics
        metrics = {
            'total_images': len(image_files),
            'total_detections': 0,
            'inference_times': [],
            'per_class': {
                cls_name: {
                    'predictions': [],
                    'ground_truths': [],
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0
                } for cls_name in self.class_names
            }
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
            self._save_visualization(img, results, output_path / f"{img_file.stem}_eval.jpg", label_file)
            
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
        """Convert YOLO results to prediction format and normalize coordinates"""
        predictions = []
        boxes = results.boxes
        img_width = results.orig_shape[1]
        img_height = results.orig_shape[0]
        
        for i in range(len(boxes)):
            box = boxes[i]
            xywh = box.xywh[0].tolist()  # x_center, y_center, width, height
            # Normalize coordinates
            normalized_box = [
                xywh[0] / img_width,  # x_center
                xywh[1] / img_height,  # y_center
                xywh[2] / img_width,  # width
                xywh[3] / img_height,  # height
            ]
            pred = {
                'class_id': int(box.cls),
                'confidence': float(box.conf),
                'box': normalized_box
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
        
        # Debug: Print coordinates
        self.logger.info(f"\nPredictions ({len(predictions)}):")
        for pred in predictions:
            self.logger.info(f"Class: {self.class_names[pred['class_id']]}, Box: {pred['box']}, Conf: {pred['confidence']:.3f}")
        
        self.logger.info(f"\nGround Truths ({len(ground_truths)}):")
        for gt in ground_truths:
            self.logger.info(f"Class: {self.class_names[gt['class_id']]}, Box: {gt['box']}")
        
        # Process each prediction
        for pred in predictions:
            metrics['total_detections'] += 1
            cls_name = self.class_names[pred['class_id']]
            metrics['per_class'][cls_name]['predictions'].append(pred)
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(ground_truths):
                if i in matched_gt or gt['class_id'] != pred['class_id']:
                    continue
                
                iou = self._calculate_iou(pred['box'], gt['box'])
                # Debug: Print IoU values
                self.logger.info(f"IoU between pred {pred['box']} and gt {gt['box']}: {iou:.3f}")
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_gt_idx >= 0:
                # True positive
                metrics['per_class'][cls_name]['true_positives'] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False positive
                metrics['per_class'][cls_name]['false_positives'] += 1
        
        # Count false negatives
        for i, gt in enumerate(ground_truths):
            cls_name = self.class_names[gt['class_id']]
            metrics['per_class'][cls_name]['ground_truths'].append(gt)
            if i not in matched_gt:
                metrics['per_class'][cls_name]['false_negatives'] += 1
    
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
        
        for cls_name, cls_metrics in metrics['per_class'].items():
            tp = cls_metrics['true_positives']
            fp = cls_metrics['false_positives']
            fn = cls_metrics['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AP using predictions and ground truths
            predictions = sorted(cls_metrics['predictions'], key=lambda x: x['confidence'], reverse=True)
            total_gt = len(cls_metrics['ground_truths'])
            
            if predictions and total_gt > 0:
                # Calculate cumulative TP and FP
                cumul_tp = 0
                cumul_fp = 0
                precisions = []
                recalls = []
                
                for pred in predictions:
                    matched = False
                    for gt in cls_metrics['ground_truths']:
                        if self._calculate_iou(pred['box'], gt['box']) >= self.iou_threshold:
                            matched = True
                            break
                    
                    if matched:
                        cumul_tp += 1
                    else:
                        cumul_fp += 1
                    
                    current_precision = cumul_tp / (cumul_tp + cumul_fp)
                    current_recall = cumul_tp / total_gt
                    
                    precisions.append(current_precision)
                    recalls.append(current_recall)
                
                # Calculate AP using precision-recall curve
                ap = 0
                for i in range(len(recalls)-1):
                    ap += (recalls[i+1] - recalls[i]) * precisions[i]
            else:
                ap = 0.0
            
            final_metrics['classes'][cls_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'ap': ap,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'total_predictions': len(predictions),
                'total_ground_truths': total_gt
            }
            
            # Store AP in the original metrics for plotting
            metrics['per_class'][cls_name]['ap'] = ap
        
        # Calculate mAP
        final_metrics['mAP'] = np.mean([
            cls_metrics['ap'] for cls_metrics in final_metrics['classes'].values()
        ])
        
        return final_metrics
    
    def _plot_metrics(self, metrics: Dict, predictions: List[Dict], ground_truths: List[Dict], output_path: Path):
        """Plot evaluation metrics"""
        # 1. Confusion Matrix
        y_true = []
        y_pred = []
        
        # Process each class separately
        for cls_name, cls_metrics in metrics['per_class'].items():
            cls_id = self.class_names.index(cls_name)
            # Add true positives
            y_true.extend([cls_id] * cls_metrics['true_positives'])
            y_pred.extend([cls_id] * cls_metrics['true_positives'])
            # Add false positives
            y_true.extend([1-cls_id] * cls_metrics['false_positives'])  # Other class
            y_pred.extend([cls_id] * cls_metrics['false_positives'])
            # Add false negatives
            y_true.extend([cls_id] * cls_metrics['false_negatives'])
            y_pred.extend([1-cls_id] * cls_metrics['false_negatives'])  # Other class
        
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
        for cls_name, cls_metrics in metrics['per_class'].items():
            if cls_metrics['predictions']:
                # Sort predictions by confidence
                sorted_preds = sorted(cls_metrics['predictions'], key=lambda x: x['confidence'], reverse=True)
                confidences = [pred['confidence'] for pred in sorted_preds]
                
                # Calculate precision and recall points
                tp = 0
                fp = 0
                total_positives = cls_metrics['true_positives'] + cls_metrics['false_negatives']
                precisions = []
                recalls = []
                
                for pred in sorted_preds:
                    if any(self._calculate_iou(pred['box'], gt['box']) >= self.iou_threshold 
                          for gt in cls_metrics['ground_truths']):
                        tp += 1
                    else:
                        fp += 1
                    
                    precision = tp / (tp + fp)
                    recall = tp / total_positives if total_positives > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                plt.plot(recalls, precisions, label=f'{cls_name} (AP={cls_metrics["ap"]:.3f})')
        
        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / 'precision_recall_curves.png')
        plt.close()
        
        # 3. Confidence Distribution
        plt.figure(figsize=(10, 8))
        for cls_name, cls_metrics in metrics['per_class'].items():
            if cls_metrics['predictions']:
                confidences = [pred['confidence'] for pred in cls_metrics['predictions']]
                plt.hist(confidences, bins=20, alpha=0.5, label=cls_name)
        
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
    
    def _save_visualization(self, img: np.ndarray, results, output_path: Path, label_file: Path):
        """Save detection visualization with both predictions and ground truth"""
        # Create copies for visualization
        orig_with_gt = img.copy()
        pred_img = img.copy()
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Draw ground truth boxes on original image
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    cls_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Draw ground truth box
                    cv2.rectangle(orig_with_gt, (x1, y1), (x2, y2), self.gt_color, self.thickness)
                    # Add class label
                    label = self.class_names[int(cls_id)]
                    (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
                    cv2.rectangle(orig_with_gt, (x1, y1-text_h-5), (x1+text_w, y1), self.gt_color, -1)
                    cv2.putText(orig_with_gt, label, (x1, y1-5), self.font, self.font_scale, (255,255,255), 1)
        
        # Draw predictions with custom color
        boxes = results.boxes
        for box in boxes:
            if box.conf >= self.conf_threshold:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Draw prediction box
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), self.pred_color, self.thickness)
                
                # Add class label and confidence
                label = f"{self.class_names[int(box.cls)]} {float(box.conf):.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
                cv2.rectangle(pred_img, (x1, y1-text_h-5), (x1+text_w, y1), self.pred_color, -1)
                cv2.putText(pred_img, label, (x1, y1-5), self.font, self.font_scale, (255,255,255), 1)
        
        # Create side-by-side comparison
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = orig_with_gt  # Original image with ground truth
        comparison[:, w:] = pred_img  # Predictions
        
        # Add text labels
        cv2.putText(comparison, 'Ground Truth', (10, 30), self.font, 1, self.gt_color, 2)
        cv2.putText(comparison, 'Predictions', (w+10, 30), self.font, 1, self.pred_color, 2)
        
        # Save the comparison
        cv2.imwrite(str(output_path), comparison)

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
    
    if metrics is None:
        print("Evaluation failed!")
        return
    
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
        print(f"  True Positives: {cls_metrics['true_positives']}")
        print(f"  False Positives: {cls_metrics['false_positives']}")
        print(f"  False Negatives: {cls_metrics['false_negatives']}")
        print(f"  Total Predictions: {cls_metrics['total_predictions']}")
        print(f"  Total Ground Truths: {cls_metrics['total_ground_truths']}")

if __name__ == '__main__':
    main() 