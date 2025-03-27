from ultralytics import YOLO
import torch
import numpy as np
from .augmentation import PCBAugmentation

class DefectDetector:
    def __init__(self, model_path="yolov8x-seg.pt", settings=None):
        self.settings = settings
        self.model = YOLO(model_path)
        self.aug = PCBAugmentation()
        self.conf_thresh = 0.5  # Default confidence threshold
        
    def detect(self, image):
        """Detect defects in the image"""
        # Run inference
        results = self.model(image)
        
        # Post-process results
        processed_results = self._process_results(results[0])
        
        return processed_results
    
    def _process_results(self, result):
        """Process YOLO results and apply thresholds"""
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
            if score < self.conf_thresh:
                continue
                
            cls_name = self.model.names[int(cls)]
            defect = {
                'class': cls_name,
                'confidence': float(score),
                'bbox': [float(x) for x in box.tolist()],
                'measurements': self._compute_measurements(
                    box, masks[idx] if masks is not None else None
                )
            }
            
            processed_defects.append(defect)
            
        return processed_defects
    
    def _compute_measurements(self, box, mask=None):
        """Compute physical measurements of defects"""
        x1, y1, x2, y2 = map(float, box)
        
        measurements = {
            'area_pixels': float((x2 - x1) * (y2 - y1)),
            'width_mm': float((x2 - x1) * self.settings.PIXEL_TO_MM_RATIO if self.settings else 0.1),
            'height_mm': float((y2 - y1) * self.settings.PIXEL_TO_MM_RATIO if self.settings else 0.1)
        }
        
        if mask is not None:
            measurements['mask_area'] = float(np.sum(mask))
            
        return measurements 