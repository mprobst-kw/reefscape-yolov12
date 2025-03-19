#!/usr/bin/env python3
# Inference script for YOLOv12 on a single image with JSON output

import argparse
import json
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv12 inference on a single image and output JSON")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (e.g., best.pt or last.pt)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to run inference on')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Size of input images as integer (default: 640)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the JSON output (default: same name as input with .json extension)')
    parser.add_argument('--include-image-path', action='store_true',
                        help='Include the image path in the JSON output')
    parser.add_argument('--normalized-coords', action='store_true',
                        help='Output normalized coordinates (0-1) instead of pixel values')
    parser.add_argument('--save-visualization', action='store_true',
                        help='Save a visualization of the detections')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.with_suffix('.json')
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"MPS (Metal Performance Shaders) is available! Using Mac GPU acceleration.")
    else:
        device = "cpu"
        print(f"MPS is not available. Using CPU.")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on {image_path}...")
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        save=args.save_visualization,
        imgsz=args.imgsz,
        device=device,
        verbose=False
    )[0]  # Get the first (and only) result
    
    # Extract detections
    detections = []
    
    # Get image dimensions for normalization if needed
    if args.normalized_coords and hasattr(results, 'orig_shape'):
        img_height, img_width = results.orig_shape
    else:
        img_height, img_width = None, None
    
    # Process detection boxes
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for i, box in enumerate(results.boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Normalize coordinates if requested
            if args.normalized_coords and img_height and img_width:
                x1 /= img_width
                x2 /= img_width
                y1 /= img_height
                y2 /= img_height
            
            # Get class info
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            }
            
            detections.append(detection)
    
    # Create final output JSON
    output_json = {
        "num_detections": len(detections),
        "detections": detections
    }
    
    # Add image path if requested
    if args.include_image_path:
        output_json["image_path"] = str(image_path)
    
    # Add metadata
    output_json["metadata"] = {
        "model": str(model_path),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "image_size": args.imgsz,
        "normalized_coordinates": args.normalized_coords
    }
    
    # Save JSON output
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"Detections saved to {output_path}")
    print(f"Found {len(detections)} objects:")
    
    # Print a simple summary
    class_counts = {}
    for det in detections:
        cls_name = det["class_name"]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")

if __name__ == "__main__":
    main() 