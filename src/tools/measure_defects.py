"""
Measure specific characteristics of PCB defects.

Measurement Metrics Explained:

For Solder Paste Deposits (Bad_slope):
- aspect_ratio: Ratio of width to height of the deposit
    * Industry standard: 1.5:1 to 1.8:1 (width:height)
    * >2.0:1 may indicate smearing or spreading issues
    * <1.5:1 might suggest insufficient paste volume
    * Critical for proper paste release from stencil

- coverage_percentage: Percentage of ROI covered by solder paste
    * Indicates the amount of pad area covered
    * Helps identify insufficient or excessive paste volume
    * Calculated using Otsu's thresholding to separate paste from background

- approximate_slope_angle: Estimated slope from intensity gradients
    * WARNING: This is an approximation from 2D image
    * Based on image intensity changes, not actual physical slope
    * Should be used as relative indicator only
    * True slope measurement requires 3D/side-view data

- intensity_std: Standard deviation of intensity values
    * Indicates paste deposit uniformity
    * Higher values suggest more irregular deposit
    * Affected by lighting conditions

For Solder Bridges (Bad_bridge):
- bridge_aspect_ratio: Ratio of bridge width to length
    * Higher ratio indicates wider bridge (more severe)
    * Lower ratio suggests thin bridge (potentially weaker)
    * Helps characterize bridge geometry

- bridge_count: Number of distinct bridges detected
    * Multiple bridges indicate systematic issues
    * Based on contour analysis with minimum area threshold

- intensity_range: Difference between max and min intensity
    * Indicates bridge completeness
    * Larger range suggests varying bridge thickness
    * Helps identify partial vs complete bridges

Common Metrics:
- width_pixels, height_pixels: Actual dimensions in pixels
    * Raw measurements for size comparison
    * Used for aspect ratio calculation
    * Useful for tracking defect size distribution

- area_pixels: Total area of the defect
    * Indicates overall defect size
    * Useful for defect severity assessment
    * Base metric for size-based filtering

Note: All intensity-based measurements are affected by:
- Lighting conditions
- Surface reflectivity
- Camera angle and settings
- Solder paste properties
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import math

class DefectMeasurement:
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('DefectMeasurement')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger
    
    def measure_defect(self, image_path: str, bbox: List[float], defect_type: str) -> Dict:
        """Measure specific characteristics of a defect"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert bbox from YOLO format to pixel coordinates
        x_center, y_center, width, height = bbox
        img_h, img_w = img.shape[:2]
        x1 = int((x_center - width/2) * img_w)
        y1 = int((y_center - height/2) * img_h)
        x2 = int((x_center + width/2) * img_w)
        y2 = int((y_center + height/2) * img_h)
        
        # Extract ROI
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            raise ValueError("Invalid bounding box coordinates")
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        measurements = {}
        if defect_type == "Bad_slope":
            measurements = self._measure_slope(gray_roi, (x1, y1, x2, y2))
        elif defect_type == "Bad_bridge":
            measurements = self._measure_bridging(gray_roi, (x1, y1, x2, y2))
        
        return measurements
    
    def _measure_slope(self, roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Measure solder paste slope characteristics.
        
        Note: Slope angle calculation from top-down view has limitations:
        1. True physical slope cannot be accurately determined without:
           - Side-view images
           - 3D height information
           - Structured light/laser scanning data
        2. Current implementation uses intensity gradients which:
           - Only indicate intensity changes in the image
           - Are affected by lighting conditions
           - May not correspond to actual physical slope
        3. Results should be used as relative indicators only
        """
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        
        # Calculate geometric properties
        aspect_ratio = width / height if height != 0 else 0
        area = width * height
        
        # Calculate intensity-based properties
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        
        # Calculate coverage percentage
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coverage = np.sum(binary > 0) / (width * height)
        
        # Calculate approximate slope indicators (with limitations)
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        # Filter strong edges
        strong_edges = magnitude > np.percentile(magnitude, 90)
        edge_directions = direction[strong_edges]
        
        # Calculate slope statistics
        avg_slope = np.mean(np.abs(edge_directions)) if edge_directions.size > 0 else 0
        std_slope = np.std(np.abs(edge_directions)) if edge_directions.size > 0 else 0
        
        return {
            'width_pixels': float(width),
            'height_pixels': float(height),
            'aspect_ratio': float(aspect_ratio),
            'area_pixels': float(area),
            'mean_intensity': float(mean_intensity),
            'intensity_std': float(std_intensity),
            'coverage_percentage': float(coverage * 100),
            'approximate_slope_angle': float(avg_slope),
            'slope_std_dev': float(std_slope),
            'note': 'Slope measurements are approximate indicators only. True physical slope cannot be determined from top-down view.'
        }
    
    def _measure_bridging(self, roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Measure solder bridging characteristics"""
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate bridge dimensions
        bridge_width = xmax - xmin
        bridge_length = ymax - ymin
        bridge_area = bridge_width * bridge_length
        aspect_ratio = bridge_width / bridge_length if bridge_length != 0 else 0
        
        # Calculate intensity profile across the bridge
        intensity_profile = np.mean(roi, axis=0)  # average intensity along columns
        min_intensity = np.min(intensity_profile)
        max_intensity = np.max(intensity_profile)
        intensity_range = max_intensity - min_intensity
        
        # Apply adaptive thresholding for bridge analysis
        thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Count significant bridges
        min_area = 50  # Minimum area to consider as a bridge
        bridge_count = sum(1 for c in contours if cv2.contourArea(c) > min_area)
        
        return {
            'bridge_width_pixels': float(bridge_width),
            'bridge_length_pixels': float(bridge_length),
            'bridge_area_pixels': float(bridge_area),
            'bridge_aspect_ratio': float(aspect_ratio),
            'min_intensity': float(min_intensity),
            'max_intensity': float(max_intensity),
            'intensity_range': float(intensity_range),
            'bridge_count': bridge_count
        }
    
    def visualize_measurements(self, image_path: str, 
                             bbox: List[float], 
                             defect_type: str,
                             measurements: Dict,
                             output_path: Optional[str] = None) -> None:
        """Visualize defect measurements"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert bbox from YOLO format to pixel coordinates
        x_center, y_center, width, height = bbox
        img_h, img_w = img.shape[:2]
        x1 = int((x_center - width/2) * img_w)
        y1 = int((y_center - height/2) * img_h)
        x2 = int((x_center + width/2) * img_w)
        y2 = int((y_center + height/2) * img_h)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add measurements text
        y_text = y1 - 10
        if defect_type == "Bad_slope":
            cv2.putText(img, 
                       f"Aspect Ratio: {measurements['aspect_ratio']:.2f}",
                       (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_text -= 20
            cv2.putText(img, 
                       f"Coverage: {measurements['coverage_percentage']:.1f}%",
                       (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_text -= 20
            cv2.putText(img, 
                       f"~Slope: {measurements['approximate_slope_angle']:.1f}°*",
                       (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:  # Bad_bridge
            cv2.putText(img, 
                       f"Bridge Width: {measurements['bridge_width_pixels']:.1f}px",
                       (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_text -= 20
            cv2.putText(img, 
                       f"Aspect Ratio: {measurements['bridge_aspect_ratio']:.2f}",
                       (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_text -= 20
            cv2.putText(img, 
                       f"Bridge Count: {measurements['bridge_count']}",
                       (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, img)
        else:
            cv2.imshow('Measurements', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure PCB defect characteristics')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--bbox', type=float, nargs=4, required=True,
                      help='Bounding box coordinates (x_center y_center width height)')
    parser.add_argument('--type', type=str, required=True,
                      choices=['Bad_slope', 'Bad_bridge'],
                      help='Type of defect')
    parser.add_argument('--output', type=str,
                      help='Path to save visualization')
    args = parser.parse_args()
    
    measurer = DefectMeasurement()
    measurements = measurer.measure_defect(args.image, args.bbox, args.type)
    
    print(f"\nMeasurements for {args.type}:")
    for key, value in measurements.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    measurer.visualize_measurements(
        args.image, args.bbox, args.type, 
        measurements, args.output
    )

if __name__ == '__main__':
    main() 