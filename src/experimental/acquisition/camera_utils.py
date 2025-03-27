import pypylon.pylon as pylon
import numpy as np
import cv2

class CameraSetup:
    def __init__(self, settings):
        self.settings = settings
        self._initialize_camera()
        
    def _initialize_camera(self):
        # Initialize Basler camera
        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        
        # Apply settings
        self.camera.Width = self.settings.CAMERA_SETTINGS["resolution"][0]
        self.camera.Height = self.settings.CAMERA_SETTINGS["resolution"][1]
        self.camera.ExposureTime = self.settings.CAMERA_SETTINGS["exposure_time"]
        self.camera.Gain = self.settings.CAMERA_SETTINGS["gain"]
        
    def get_camera_instance(self):
        return self.camera 