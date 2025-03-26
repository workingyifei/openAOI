import numpy as np
from scipy.spatial import distance

class DefectMetrics:
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