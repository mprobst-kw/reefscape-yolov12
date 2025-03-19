#!/usr/bin/env python3
# Inference script for YOLOv12 on new coral reef images

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

def parse_args():
    # Determine best available device
    if torch.backends.mps.is_available():
        default_device = "mps"  # Use Metal Performance Shaders for Mac M-series
    else:
        default_device = "cpu"
        
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLOv12 model")
    parser.add_argument('--model', type=str, default='runs/detect/reefscape_yolov12/weights/best.pt',
                        help='Path to the trained YOLOv12 model')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the image or directory of images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=300,
                        help='Maximum number of detections per image')
    parser.add_argument('--device', type=str, default=default_device,
                        help='Device to run inference on (e.g., "mps" for Mac GPU, "cpu" for CPU)')
    parser.add_argument('--save-dir', type=str, default='runs/detect/predict_results',
                        help='Directory to save results')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--view-img', action='store_true',
                        help='Show results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Print device info
    if args.device == "mps":
        print("Using Mac M-series GPU acceleration with Metal Performance Shaders (MPS)")
    elif args.device == "cpu":
        print("Using CPU for inference")
    else:
        print(f"Using device: {args.device}")
    
    # Load trained model
    try:
        model = YOLO(args.model)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get source path
    source = Path(args.source)
    if not source.exists():
        print(f"Source {source} does not exist!")
        return
    
    # Run inference
    results = model.predict(
        source=str(source),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        imgsz=args.imgsz,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        project=save_dir.parent,
        name=save_dir.name
    )
    
    # Display and save results
    print(f"\nResults saved to {save_dir}")
    
    # If view_img flag is set, display detections
    if args.view_img:
        for i, res in enumerate(results):
            # Get the annotated image with detections
            annotated_img = res.plot()
            
            # Get class counts
            class_counts = {}
            for box in res.boxes:
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            class_info = ", ".join([f"{count} {name}" for name, count in class_counts.items()])
            
            # Display image with detections
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_img)
            plt.title(f"Detections: {class_info}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
    
    # Print summary of detections
    total_detections = sum(len(res.boxes) for res in results)
    print(f"Total detections: {total_detections}")
    
    # Print class distribution
    class_counts = {}
    for res in results:
        for box in res.boxes:
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    print("\nClass distribution:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count} ({count/total_detections*100:.1f}%)")
    
    # Calculate and print average confidence
    confidences = [float(box.conf) for res in results for box in res.boxes]
    avg_conf = np.mean(confidences) if confidences else 0
    print(f"\nAverage confidence: {avg_conf:.4f}")
    
    print(f"\nInference complete. Results saved to {save_dir}")

if __name__ == "__main__":
    main() 