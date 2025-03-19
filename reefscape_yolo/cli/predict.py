#!/usr/bin/env python3
"""Run inference with a trained YOLOv12 model on the REEFSCAPE dataset."""

import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import time
import random
import yaml

from reefscape_yolo.core.model import load_model
from reefscape_yolo.core.inference import process_image
from reefscape_yolo.utils.image import draw_detections
from reefscape_yolo.core.dataset import find_dataset_yaml, setup_dataset_symlink
from reefscape_yolo.config.defaults import INFERENCE_DEFAULTS, DEFAULT_DATASET


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with YOLOv12 on the REEFSCAPE dataset")
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory with images or path to a single image')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--device', type=str, default=INFERENCE_DEFAULTS['device'],
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--imgsz', type=int, default=INFERENCE_DEFAULTS['imgsz'],
                        help=f'Size of input images as integer (default: {INFERENCE_DEFAULTS["imgsz"]})')
    parser.add_argument('--conf', type=float, default=INFERENCE_DEFAULTS['conf'],
                        help=f'Confidence threshold (default: {INFERENCE_DEFAULTS["conf"]})')
    parser.add_argument('--iou', type=float, default=INFERENCE_DEFAULTS['iou'],
                        help=f'IoU threshold for NMS (default: {INFERENCE_DEFAULTS["iou"]})')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to *.json')
    parser.add_argument('--save-img', action='store_true',
                        help='Save annotated images')
    parser.add_argument('--project', type=str, default=INFERENCE_DEFAULTS['project'],
                        help=f'Save results to project/name (default: {INFERENCE_DEFAULTS["project"]})')
    parser.add_argument('--name', type=str, default=INFERENCE_DEFAULTS['name'],
                        help=f'Save results to project/name (default: {INFERENCE_DEFAULTS["name"]})')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of images to process (for testing)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for image selection if limiting')
    parser.add_argument('--setup-dataset', action='store_true',
                        help='Setup dataset symlink to standard location')
    parser.add_argument('--data', type=str, default=DEFAULT_DATASET,
                        help=f'Path to dataset configuration (only used to find test directory) (default: {DEFAULT_DATASET})')
    return parser.parse_args()


def find_images(source):
    """Find images in the source directory or return a single image path."""
    source = Path(source)
    
    # If source is "test", try to find the test directory in the dataset
    if source.name == "test" and not source.exists():
        dataset_yaml = find_dataset_yaml()
        if dataset_yaml:
            with open(dataset_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Get the test path from the YAML
            test_dir = dataset_yaml.parent / data.get('test', 'test/images')
            if test_dir.exists():
                print(f"Using test directory from dataset: {test_dir}")
                source = test_dir
    
    if source.is_dir():
        # Look for image files in the directory
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        images = []
        for ext in exts:
            images.extend(list(source.glob(f'*{ext}')))
            images.extend(list(source.glob(f'*{ext.upper()}')))
        
        return sorted(images)
    else:
        # Single image file
        if source.exists():
            return [source]
        else:
            raise ValueError(f"Source file not found: {source}")


def main():
    args = parse_args()
    
    # Setup dataset symlink if requested
    if args.setup_dataset:
        setup_dataset_symlink()
    
    # Check if weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Model weights not found: {weights_path}")
        return
    
    # Load model
    print(f"Loading YOLOv12 model from {args.weights}...")
    model = load_model(args.weights, args.device)
    
    # Find images
    print(f"Finding images in {args.source}...")
    try:
        images = find_images(args.source)
        print(f"Found {len(images)} images")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Limit images if requested
    if args.limit and args.limit < len(images):
        if args.seed is not None:
            random.seed(args.seed)
        
        images = random.sample(images, args.limit)
        print(f"Limited to {len(images)} images")
    
    if not images:
        print("No images found. Exiting.")
        return
    
    # Create output directory
    save_dir = Path(args.project) / args.name
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}")
    
    # Process images
    total_time = 0
    total_images = len(images)
    
    for i, img_path in enumerate(images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error loading image {img_path}. Skipping.")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.time()
        results = process_image(
            model=model, 
            image_data=img_rgb, 
            conf=args.conf, 
            iou=args.iou, 
            img_size=args.imgsz
        )
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Print progress
        print(f"[{i+1}/{total_images}] {img_path.name}: {len(results['detections'])} detections ({inference_time:.3f}s)")
        
        # Save results
        if args.save_json:
            json_path = save_dir / f"{img_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        if args.save_txt:
            txt_path = save_dir / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for det in results['detections']:
                    bbox = det['bbox']
                    line = f"{det['class_id']} {bbox['x1']} {bbox['y1']} {bbox['x2']} {bbox['y2']}"
                    if args.save_conf:
                        line += f" {det['confidence']:.6f}"
                    f.write(line + '\n')
        
        if args.save_img:
            # Draw detections on image
            img_with_dets = draw_detections(img_rgb, results['detections'])
            img_with_dets = cv2.cvtColor(img_with_dets, cv2.COLOR_RGB2BGR)
            img_save_path = save_dir / f"{img_path.stem}_det.jpg"
            cv2.imwrite(str(img_save_path), img_with_dets)
    
    # Print summary
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"\nProcessed {total_images} images in {total_time:.2f}s ({avg_time:.3f}s/image)")
        print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main() 