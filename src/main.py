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
from models.detector import DefectDetector
from qa.ipc_checker import IPC610Validator

def load_image(image_path):
    """Load and validate an input image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img

def main():
    parser = argparse.ArgumentParser(description='OpenAOI Inspection System')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                      help='Path to YOLO model file')
    parser.add_argument('--test-image', type=str, required=True,
                      help='Path to test image')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize detector and validator
    detector = DefectDetector(model_path=args.model)
    validator = IPC610Validator(None)  # No settings needed for now
    
    # Load and process image
    try:
        image = load_image(args.test_image)
        
        # Detect defects
        defects = detector.detect(image)
        
        # Validate defects
        validated_defects = validator.validate(defects)
        
        # Save results
        results = {
            'image_path': args.test_image,
            'total_defects': len(validated_defects),
            'defects': validated_defects
        }
        
        output_file = os.path.join(args.output_dir, 'results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\nInspection Results:")
        print(f"Total defects found: {len(validated_defects)}")
        print(f"Results saved to: {output_file}")
        
        if validated_defects:
            print("\nDetected Defects:")
            for defect in validated_defects:
                print(f"- {defect['class']}: {defect.get('confidence', 0):.2f} " +
                      f"({defect['ipc_status']})")
        
    except Exception as e:
        print(f"Error during inspection: {str(e)}")
        raise

if __name__ == '__main__':
    main() 