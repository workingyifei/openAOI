from ultralytics import YOLO
import torch
import numpy as np
from .augmentation import PCBAugmentation

class DefectDetector:
    def __init__(self, model_path="yolov8x-seg.pt", settings=None):
        self.settings = settings
        self.model = YOLO(model_path)
        self.aug = PCBAugmentation()
        self.class_thresh = self._load_class_thresholds()
        
    def _load_class_thresholds(self):
        """Load class-specific thresholds from IPC standards"""
        ipc_classes = self.settings.load_ipc_classes()
        thresholds = {}
        for defect in ipc_classes['defect_classes']:
            thresholds[defect['name']] = {
                'confidence': 0.5,  # Default confidence threshold
                **defect['class3_limits']
            }
        return thresholds
    
    def detect(self, image):
        """Detect defects in the image"""
        # Augment image for inference
        aug_data = self.aug(image=image)
        aug_image = aug_data['image']
        
        # Run inference
        results = self.model(aug_image)
        
        # Post-process results
        processed_results = self._process_results(results[0])
        
        return processed_results
    
    def _process_results(self, result):
        """Process YOLO results and apply IPC thresholds"""
        processed_defects = []
        
        # Extract boxes, scores, and class predictions
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        # If segmentation masks are available
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
        else:
            masks = None
        
        for idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            cls_name = self.model.names[int(cls)]
            
            # Apply class-specific threshold
            if score < self.class_thresh[cls_name]['confidence']:
                continue
            
            defect = {
                'class': cls_name,
                'confidence': float(score),
                'bbox': box.tolist(),
                'measurements': self._compute_measurements(
                    box, masks[idx] if masks is not None else None
                )
            }
            
            processed_defects.append(defect)
            
        return processed_defects
    
    def _compute_measurements(self, box, mask=None):
        """Compute physical measurements of defects"""
        x1, y1, x2, y2 = box
        
        measurements = {
            'area_pixels': (x2 - x1) * (y2 - y1),
            'width_mm': (x2 - x1) * self.settings.PIXEL_TO_MM_RATIO,
            'height_mm': (y2 - y1) * self.settings.PIXEL_TO_MM_RATIO,
        }
        
        if mask is not None:
            measurements['mask_area'] = np.sum(mask)
            
        return measurements 