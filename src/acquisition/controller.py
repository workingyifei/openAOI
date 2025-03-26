from .camera_utils import CameraSetup
import numpy as np
import cv2

class InspectionCamera:
    def __init__(self, settings):
        self.settings = settings
        self.camera_setup = CameraSetup(settings)
        self.camera = self.camera_setup.get_camera_instance()
        
    def capture_stack(self, angles=None):
        """Capture images at multiple angles with polarization control"""
        if angles is None:
            angles = self.settings.INSPECTION_ANGLES
            
        images = []
        for angle in angles:
            self._rotate_stage(angle)
            images.append(self._capture_with_polarization())
        return images
    
    def _capture_with_polarization(self):
        """Capture with different polarization filters"""
        polarized_images = []
        for pol_angle in [0, 45, 90, 135]:
            self._set_polarization(pol_angle)
            img = self._capture_single()
            polarized_images.append(img)
        
        # Compute polarization features
        return self._compute_polarization_features(polarized_images)
    
    def _capture_single(self):
        """Capture a single frame"""
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        with self.camera.RetrieveResult(5000) as result:
            if result.GrabSucceeded():
                img = result.Array
                return cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
        return None
    
    def _rotate_stage(self, angle):
        """Control the rotation stage - implement hardware specific control"""
        pass
    
    def _set_polarization(self, angle):
        """Control the polarization filter - implement hardware specific control"""
        pass
    
    def _compute_polarization_features(self, images):
        """Compute polarization features from multiple polarized images"""
        # Stokes parameters calculation
        I = np.sum(images, axis=0) / 4
        Q = (images[0] - images[2]) / 2
        U = (images[1] - images[3]) / 2
        
        # Degree of polarization
        DoP = np.sqrt(Q**2 + U**2) / I
        
        return {
            'intensity': I,
            'dop': DoP,
            'polarization_angle': 0.5 * np.arctan2(U, Q)
        } 