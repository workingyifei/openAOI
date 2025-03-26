import albumentations as A
import cv2

class PCBAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.RandomBrightness(limit=0.2, p=1),
                A.RandomContrast(limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1)
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), p=1)
            ], p=0.5),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.3
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def __call__(self, image, bboxes=None, class_labels=None):
        if bboxes is None:
            return self.transform(image=image)
        return self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        ) 