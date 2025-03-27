import numpy as np
from scipy.spatial import distance

class DefectMetrics:
    @staticmethod
    def compute_bridge_metrics(defect, pixel_to_mm):
        """Compute metrics for solder bridging (SolderBridgingDefect)"""
        bbox = np.array(defect['bbox'])
        width_mm = float((bbox[2] - bbox[0]) * pixel_to_mm)
        height_mm = float((bbox[3] - bbox[1]) * pixel_to_mm)
        
        return {
            'bridge_width': width_mm,
            'bridge_height': height_mm,
            'area_mm2': float(width_mm * height_mm)
        }
    
    @staticmethod
    def compute_slope_metrics(defect, mask):
        """Compute metrics for solder paste slope (SolderSlopeDefect)"""
        if mask is None:
            return {}
            
        # Calculate the slope angle using the mask
        # This is a simplified version - in practice you might want to use
        # more sophisticated edge detection and line fitting
        edges = np.where(mask > 0)
        if len(edges[0]) < 2:
            return {}
            
        # Fit a line to the edge points
        x = edges[1]
        y = edges[0]
        slope = float(np.polyfit(x, y, 1)[0])
        angle = float(np.abs(np.arctan(slope) * 180 / np.pi))
        
        return {
            'slope_angle': angle,
            'slope': slope
        }
    
    @staticmethod
    def compute_solder_bridge_metrics(defect, pixel_to_mm):
        """Compute metrics for solder bridging"""
        bbox = np.array(defect['bbox'])
        width_mm = (bbox[2] - bbox[0]) * pixel_to_mm
        height_mm = (bbox[3] - bbox[1]) * pixel_to_mm
        
        return {
            'max_distance': max(width_mm, height_mm),
            'area_mm2': width_mm * height_mm
        }
    
    @staticmethod
    def compute_void_metrics(defect, mask):
        """Compute metrics for solder voiding"""
        if mask is None:
            return {}
            
        total_area = np.sum(mask)
        void_area = np.sum(mask == 0)
        void_percentage = (void_area / total_area) * 100
        
        return {
            'void_percentage': void_percentage,
            'total_area': total_area,
            'void_area': void_area
        }
    
    @staticmethod
    def compute_tombstone_metrics(defect, point_cloud):
        """Compute metrics for tombstoning"""
        if point_cloud is None:
            return {}
            
        # Fit plane to component surface
        plane = DefectMetrics._fit_plane(point_cloud)
        angle = DefectMetrics._compute_angle(plane)
        
        return {
            'angle_degrees': angle,
            'height_variation': np.ptp(point_cloud[:, 2])
        }
    
    @staticmethod
    def _fit_plane(points):
        """Fit 3D plane to points"""
        centroid = np.mean(points, axis=0)
        shifted = points - centroid
        _, _, vh = np.linalg.svd(shifted)
        return vh[2]
    
    @staticmethod
    def _compute_angle(plane_normal):
        """Compute angle between plane normal and vertical"""
        vertical = np.array([0, 0, 1])
        cos_angle = np.dot(plane_normal, vertical)
        return np.arccos(cos_angle) * 180 / np.pi 