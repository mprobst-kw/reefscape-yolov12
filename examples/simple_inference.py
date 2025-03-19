#!/usr/bin/env python3
"""
Simple inference example using the REEFSCAPE YOLOv12 library.

This example shows how to:
1. Load a trained model
2. Run inference on a single image
3. Display the results
4. Save the annotated image
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports from reefscape_yolo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reefscape_yolo.core.model import load_model
from reefscape_yolo.core.inference import process_image
from reefscape_yolo.utils.image import load_image, draw_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference example")
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the model weights file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the annotated image (optional)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if files exist
    image_path = Path(args.image)
    weights_path = Path(args.weights)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not weights_path.exists():
        print(f"Error: Model weights file not found: {weights_path}")
        return
    
    # Load the model
    print(f"Loading model from {weights_path}...")
    model = load_model(weights_path)
    
    # Load the image
    print(f"Loading image from {image_path}...")
    image = load_image(str(image_path))
    
    # Run inference
    print("Running inference...")
    results = process_image(model, image, conf=args.conf, iou=args.iou)
    
    # Print detection results
    print(f"\nFound {results['num_detections']} detections:")
    for i, det in enumerate(results['detections']):
        print(f"  {i+1}. {det['class_name']} ({det['confidence']:.2f})")
    
    # Draw the detections on the image
    annotated_image = draw_detections(image, results['detections'])
    
    # Save the annotated image if requested
    if args.output:
        output_path = Path(args.output)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr_image)
        print(f"Saved annotated image to {output_path}")
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title(f"REEFSCAPE YOLOv12 Detections: {results['num_detections']} objects found")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 