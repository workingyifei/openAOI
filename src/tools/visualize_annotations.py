"""
Visualization tool for verifying annotations and displaying defect measurements.

The tool works with PCB/A inspection images captured using industrial AOI systems.
Image characteristics:
- High-resolution RGB images
- Controlled lighting conditions with angled illumination
- Multiple viewing angles for enhanced defect visibility
- Possible IR components for better solder joint visualization

Note: While the system is designed to work with live camera feeds,
the current implementation works with pre-captured images only.
"""
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
import math
from typing import Tuple, Dict, List, Set
import logging

class LabelManager:
    def __init__(self, img_height: int, img_width: int):
        self.occupied_regions = set()  # Set of occupied pixel regions
        self.img_height = img_height
        self.img_width = img_width
        self.margin = 5  # Margin between labels
        
    def find_label_position(self, bbox: Tuple[int, int, int, int], 
                          label_size: Tuple[int, int]) -> Tuple[int, int]:
        """Find an unoccupied position for the label"""
        xmin, ymin, xmax, ymax = bbox
        text_width, text_height = label_size
        
        # Try positions in this order: top, bottom, right, left
        positions = [
            # Top
            (xmin, ymin - text_height - 10),
            # Bottom
            (xmin, ymax + 10),
            # Right
            (xmax + 10, ymin),
            # Left
            (xmin - text_width - 10, ymin)
        ]
        
        for x, y in positions:
            region = self._get_region(x, y, text_width, text_height)
            if self._is_valid_position(region):
                self.occupied_regions.add(region)
                return x, y
                
        # If no good position found, try offset positions
        offset = 20
        while offset < 100:  # Limit the search radius
            x = xmin + offset
            y = ymin - offset
            region = self._get_region(x, y, text_width, text_height)
            if self._is_valid_position(region):
                self.occupied_regions.add(region)
                return x, y
            offset += 20
            
        # Fallback position
        return xmax + 5, ymin
    
    def _get_region(self, x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert label position and size to a region tuple"""
        return (x - self.margin, y - self.margin, 
                x + width + self.margin, y + height + self.margin)
    
    def _is_valid_position(self, region: Tuple[int, int, int, int]) -> bool:
        """Check if the region is valid and unoccupied"""
        x1, y1, x2, y2 = region
        
        # Check image boundaries
        if (x1 < 0 or y1 < 0 or 
            x2 >= self.img_width or 
            y2 >= self.img_height):
            return False
            
        # Check for overlaps with existing labels
        for occupied in self.occupied_regions:
            if self._regions_overlap(region, occupied):
                return False
                
        return True
    
    def _regions_overlap(self, r1: Tuple[int, int, int, int], 
                        r2: Tuple[int, int, int, int]) -> bool:
        """Check if two regions overlap"""
        return not (r1[2] < r2[0] or r1[0] > r2[2] or 
                   r1[3] < r2[1] or r1[1] > r2[3])

class AnnotationVisualizer:
    def __init__(self):
        self.colors = {
            'Bad_slope': (0, 255, 255),    # Yellow text for slope defects
            'Bad_bridge': (255, 0, 255)     # Magenta text for bridging defects
        }
        self.box_color = (255, 255, 255)    # White boxes for all defects
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('Visualization')
        logger.setLevel(logging.DEBUG)  # Set to DEBUG level
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger
    
    def visualize_dataset(self, dataset_path: str, output_path: str = 'visualization'):
        """Visualize all annotations in the dataset"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        self.logger.debug(f"Looking for images in: {dataset_path}/train_data/JPEGImages/")
        
        # Get all image files
        image_files = sorted(dataset_path.glob('train_data/JPEGImages/*.jpeg'))
        self.logger.info(f"Found {len(list(image_files))} images")
        
        for img_file in image_files:
            self.logger.debug(f"Processing image: {img_file}")
            
            # Get corresponding annotation file
            xml_file = dataset_path / 'train_data/Annotations' / f"{img_file.stem}.xml"
            self.logger.debug(f"Looking for annotation file: {xml_file}")
            
            if not xml_file.exists():
                self.logger.warning(f"Annotation not found for {img_file}")
                continue
            
            # Load and visualize
            img = cv2.imread(str(img_file))
            if img is None:
                self.logger.warning(f"Failed to load image: {img_file}")
                continue
            
            self.logger.debug(f"Image loaded successfully: {img.shape}")
            
            # Draw annotations and measurements
            vis_img = self.visualize_single(img, xml_file)
            
            # Save visualization
            output_file = output_path / f"{img_file.stem}_annotated.jpg"
            self.logger.debug(f"Saving visualization to: {output_file}")
            
            try:
                cv2.imwrite(str(output_file), vis_img)
                self.logger.debug(f"Successfully saved: {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save visualization: {str(e)}")
            
        self.logger.info(f"Visualizations saved to {output_path}")
    
    def visualize_single(self, img: np.ndarray, xml_file: Path) -> np.ndarray:
        """Visualize annotations for a single image"""
        vis_img = img.copy()
        label_manager = LabelManager(img.shape[0], img.shape[1])
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            self.logger.debug(f"Successfully parsed XML: {xml_file}")
        except Exception as e:
            self.logger.error(f"Failed to parse XML {xml_file}: {str(e)}")
            return vis_img
        
        for obj in root.findall('object'):
            try:
                name_elem = obj.find('name')
                if name_elem is None:
                    name_elem = obj.find('n')
                
                if name_elem is None:
                    self.logger.warning(f"No name/n element found in object")
                    continue
                    
                defect_type = name_elem.text.strip()
                self.logger.debug(f"Found defect type: {defect_type}")
                
                # Convert old defect names to new ones
                if defect_type == 'Bad_podu':
                    defect_type = 'Bad_slope'
                elif defect_type == 'Bad_qiaojiao':
                    defect_type = 'Bad_bridge'
                    
                bbox = obj.find('bndbox')
                if bbox is None:
                    self.logger.warning(f"No bndbox element found for {defect_type}")
                    continue
                
                # Get coordinates
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                self.logger.debug(f"Bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")
                
                # Draw bounding box
                cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), 
                            self.box_color, 2)
                
                # Get measurements
                measurements = self._measure_defect(img, defect_type, (xmin, ymin, xmax, ymax))
                
                # Format label text
                label_lines = [f"{defect_type}:"]
                for key, value in measurements.items():
                    label_lines.append(f"{key}: {value}")
                
                # Calculate text size for all lines
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                line_height = 20
                max_width = 0
                
                for line in label_lines:
                    (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                    max_width = max(max_width, w)
                
                # Get label position
                text_height = len(label_lines) * line_height
                x, y = label_manager.find_label_position(
                    (xmin, ymin, xmax, ymax),
                    (max_width, text_height)
                )
                
                # Draw leader line if label is not directly above/below box
                if x < xmin - 10 or x > xmax + 10:
                    center_x = (xmin + xmax) // 2
                    center_y = (ymin + ymax) // 2
                    self._draw_dotted_line(vis_img, (center_x, center_y), 
                                         (x + max_width//2, y + text_height//2),
                                         self.colors[defect_type])
                
                # Draw label background and text
                for i, line in enumerate(label_lines):
                    text_y = y + (i+1) * line_height
                    (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                    
                    # Draw black background for text
                    cv2.rectangle(vis_img, 
                                (x-2, text_y-h-2),
                                (x+w+2, text_y+2),
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(vis_img, line, (x, text_y),
                              font, font_scale,
                              self.colors[defect_type], thickness)
                
            except Exception as e:
                self.logger.error(f"Error processing object: {str(e)}")
                continue
        
        return vis_img
    
    def _draw_dotted_line(self, img: np.ndarray, pt1: Tuple[int, int], 
                         pt2: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a dotted line between two points"""
        dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        pts = np.linspace(pt1, pt2, int(dist))
        
        # Draw dots every 5 pixels
        for i in range(0, len(pts), 5):
            x, y = pts[i].astype(np.int32)
            cv2.circle(img, (x, y), 1, color, -1)
    
    def _measure_defect(self, img: np.ndarray, defect_type: str, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Measure defect characteristics based on type.
        Returns a dictionary of measurements with their values.
        """
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        roi = img[ymin:ymax, xmin:xmax]
        
        if roi.size == 0:
            return {}
            
        # Convert ROI to grayscale for analysis
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
        measurements = {}
        
        if defect_type == 'Bad_slope':
            # Calculate geometric properties
            aspect_ratio = width / height if height != 0 else 0
            area = width * height
            
            # Calculate intensity-based properties
            mean_intensity = np.mean(roi)
            std_intensity = np.std(roi)
            
            # Calculate coverage percentage using Otsu's thresholding
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coverage = np.sum(binary > 0) / (width * height)
            
            # Calculate approximate slope indicators
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
            
            # Filter strong edges
            strong_edges = magnitude > np.percentile(magnitude, 90)
            edge_directions = direction[strong_edges]
            avg_slope = np.mean(np.abs(edge_directions)) if edge_directions.size > 0 else 0
            
            measurements = {
                'aspect_ratio': f"{aspect_ratio:.2f}",  # Industry std: 1.5-1.8:1
                'coverage': f"{coverage * 100:.1f}%",   # Paste coverage
                'approx_slope': f"{avg_slope:.1f}°*",   # Approximate slope angle
                'uniformity': f"{100 - std_intensity/mean_intensity*100:.1f}%" if mean_intensity > 0 else "N/A"
            }
            
        elif defect_type == 'Bad_bridge':
            # Calculate bridge characteristics
            aspect_ratio = width / height if height != 0 else 0
            
            # Calculate intensity profile
            intensity_profile = np.mean(roi, axis=0)
            intensity_range = np.max(intensity_profile) - np.min(intensity_profile)
            
            # Detect bridge count using contours
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_bridges = sum(1 for c in contours if cv2.contourArea(c) > 50)
            
            measurements = {
                'width': f"{width:.1f}px",
                'aspect_ratio': f"{aspect_ratio:.2f}",
                'bridge_count': str(significant_bridges),
                'severity': f"{intensity_range/255*100:.1f}%"  # Bridge completeness
            }
        
        return measurements
    
    def _visualize_slope(self, img: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Visualize slope direction and angle"""
        xmin, ymin, xmax, ymax = bbox
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        # Draw slope direction indicator
        angle = self._calculate_slope_angle(
            img[ymin:ymax, xmin:xmax]
        )
        length = 30
        dx = length * np.cos(angle * np.pi / 180)
        dy = length * np.sin(angle * np.pi / 180)
        
        cv2.arrowedLine(
            img,
            (center_x, center_y),
            (int(center_x + dx), int(center_y + dy)),
            (0, 255, 0), 2
        )
    
    def _visualize_bridging(self, img: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Visualize bridging characteristics"""
        xmin, ymin, xmax, ymax = bbox
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        # Draw bridge connection indicators
        cv2.line(img, (xmin, center_y), (xmax, center_y), (0, 0, 255), 1)
        cv2.circle(img, (xmin, center_y), 3, (0, 0, 255), -1)
        cv2.circle(img, (xmax, center_y), 3, (0, 0, 255), -1)

def main():
    parser = argparse.ArgumentParser(description='Visualize PCB defect annotations')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='visualization_output',
                      help='Output directory for visualizations')
    args = parser.parse_args()
    
    visualizer = AnnotationVisualizer()
    visualizer.visualize_dataset(args.dataset, args.output)

if __name__ == '__main__':
    main() 