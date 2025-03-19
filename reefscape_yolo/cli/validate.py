#!/usr/bin/env python3
"""Validate a trained YOLOv12 model on the REEFSCAPE dataset."""

import os
import argparse
import json
from pathlib import Path
import time
import yaml

from reefscape_yolo.core.model import load_model
from reefscape_yolo.core.dataset import verify_dataset, find_dataset_yaml, setup_dataset_symlink
from reefscape_yolo.config.defaults import DEFAULT_DATASET, VALIDATION_DEFAULTS


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv12 on the REEFSCAPE dataset")
    parser.add_argument('--data', type=str, default=DEFAULT_DATASET,
                        help=f'Path to dataset configuration YAML file (default: {DEFAULT_DATASET})')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--device', type=str, default=VALIDATION_DEFAULTS['device'],
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--batch-size', type=int, default=VALIDATION_DEFAULTS['batch_size'],
                        help=f'Batch size for validation (default: {VALIDATION_DEFAULTS["batch_size"]})')
    parser.add_argument('--imgsz', type=int, default=VALIDATION_DEFAULTS['imgsz'],
                        help=f'Size of input images as integer (default: {VALIDATION_DEFAULTS["imgsz"]})')
    parser.add_argument('--conf', type=float, default=VALIDATION_DEFAULTS['conf'],
                        help=f'Confidence threshold (default: {VALIDATION_DEFAULTS["conf"]})')
    parser.add_argument('--iou', type=float, default=VALIDATION_DEFAULTS['iou'],
                        help=f'IoU threshold for NMS (default: {VALIDATION_DEFAULTS["iou"]})')
    parser.add_argument('--project', type=str, default=VALIDATION_DEFAULTS['project'],
                        help=f'Save results to project/name (default: {VALIDATION_DEFAULTS["project"]})')
    parser.add_argument('--name', type=str, default=VALIDATION_DEFAULTS['name'],
                        help=f'Save results to project/name (default: {VALIDATION_DEFAULTS["name"]})')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to JSON format')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='Save hybrid version of labels')
    parser.add_argument('--plots', action='store_true',
                        help='Generate validation plots')
    parser.add_argument('--setup-dataset', action='store_true',
                        help='Setup dataset symlink to standard location')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup dataset symlink if requested
    if args.setup_dataset:
        setup_dataset_symlink()
    
    # Find the dataset if path is not explicitly provided
    if not Path(args.data).exists():
        dataset_yaml = find_dataset_yaml()
        if dataset_yaml:
            args.data = str(dataset_yaml)
            print(f"Found dataset at: {args.data}")
        else:
            print(f"ERROR: Could not find dataset at {args.data} or in standard locations.")
            print("Use --setup-dataset to create a symlink to the standard location.")
            return
    
    # Setup paths
    dataset_path = Path(args.data)
    save_dir = Path(args.project) / args.name
    os.makedirs(save_dir, exist_ok=True)
    
    # Verify dataset
    print("Verifying dataset paths...")
    if not verify_dataset(dataset_path):
        print("Dataset verification failed. Please check the paths in data.yaml")
        return
    
    # Check if weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Model weights not found: {weights_path}")
        return
    
    # Load model
    print(f"Loading YOLOv12 model from {args.weights}...")
    model = load_model(args.weights, args.device)
    
    # Run validation
    print("=" * 80)
    print(f"Starting validation on REEFSCAPE dataset")
    print(f"Using device: {args.device}, batch size: {args.batch_size}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Validate the model
        val_results = model.val(
            data=str(dataset_path.absolute()),  # Use absolute path
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            project=args.project,
            name=args.name,
            plots=args.plots,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
        )
        
        validation_time = time.time() - start_time
        
        # Save validation metrics to a file
        metrics = {
            "metrics": {
                "precision": float(val_results.box.p),
                "recall": float(val_results.box.r),
                "mAP50": float(val_results.box.map50),
                "mAP50-95": float(val_results.box.map),
                "validation_time_seconds": validation_time
            },
            "parameters": {
                "model": args.weights,
                "confidence_threshold": args.conf,
                "iou_threshold": args.iou,
                "image_size": args.imgsz
            },
            "dataset": dataset_path.name
        }
        
        # Print metrics
        print("\nValidation Results:")
        print(f"Precision: {metrics['metrics']['precision']:.4f}")
        print(f"Recall: {metrics['metrics']['recall']:.4f}")
        print(f"mAP@0.5: {metrics['metrics']['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['metrics']['mAP50-95']:.4f}")
        print(f"Validation time: {validation_time:.2f}s")
        
        # Save metrics to file
        metrics_file = save_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nValidation complete. Results saved to {save_dir}")
    
    except Exception as e:
        print(f"Error during validation: {e}")


if __name__ == "__main__":
    main() 