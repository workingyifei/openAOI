"""
Data augmentation for PCB defect detection.
"""
import albumentations as A
import cv2

class PCBAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __call__(self, image, bboxes=None, class_labels=None):
        if bboxes is None:
            return self.transform(image=image)
        return self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        ) 