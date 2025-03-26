#!/usr/bin/env python3
"""
Main entry point for the OpenAOI system.
Supports both camera-based inspection and testing with pre-captured images.
"""

import argparse
import cv2
import os
import json
from pathlib import Path
from integration.workflow import InspectionWorkflow
from integration.settings import Settings

def load_image(image_path):
    """Load and validate an input image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img

class TestImageProvider:
    """Simulates camera behavior using test images"""
    def __init__(self, image_path):
        self.image = load_image(image_path)
    
    def capture_stack(self):
        """Simulate camera capture by returning the test image"""
        return [self.image]

def main():
    parser = argparse.ArgumentParser(description='OpenAOI Inspection System')
    parser.add_argument('--config', type=str, default='config/default_settings.json',
                      help='Path to configuration file')
    parser.add_argument('--test-image', type=str,
                      help='Path to test image (overrides camera input)')
    parser.add_argument('--template', type=str,
                      help='Path to PCB template image')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    args = parser.parse_args()

    # Load settings
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    settings = Settings.load_settings(args.config)
    
    # Override settings with command line arguments
    if args.test_image:
        settings.test_image_path = args.test_image
        settings.use_camera = False
    if args.template:
        settings.template_path = args.template
    if args.output_dir:
        settings.output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(settings.output_dir, exist_ok=True)

    # Initialize workflow
    workflow = InspectionWorkflow(settings)
    
    # If using test image, replace camera with test image provider
    if not settings.use_camera:
        print(f"Using test image: {settings.test_image_path}")
        workflow.cam = TestImageProvider(settings.test_image_path)

    # Load template
    template = load_image(settings.template_path)
    
    # Run inspection
    try:
        results = workflow.run(template, save_results=True)
        print("\nInspection Results:")
        print(f"Status: {results['pass_fail']}")
        print(f"Total defects found: {results['total_defects']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        if results['defects']:
            print("\nDetected Defects:")
            for defect in results['defects']:
                print(f"- {defect['class']}: {defect.get('confidence', 0):.2f} " +
                      f"({defect['ipc_status']})")
        
        print(f"\nResults saved to: {settings.output_dir}")
        
    except Exception as e:
        print(f"Error during inspection: {str(e)}")
        raise

if __name__ == '__main__':
    main() 