#!/usr/bin/env python3
# Train and test YOLOv12 on the 2025 Reefscape dataset

import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml
import torch
import argparse
import time
import sys

# Add project root to path to allow imports from utils package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set seed for reproducibility in training
# but we'll use a different seed for test image selection
random.seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and test YOLOv12 on the 2025 Reefscape dataset")
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a specific checkpoint file (e.g., path to best.pt or last.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training (default: auto-selected based on device)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--data', type=str, default="data/reefscape/data.yaml",
                        help='Path to dataset configuration file')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Size of input images as integer')
    parser.add_argument('--weights', type=str, default="yolov12n.pt",
                        help='Initial weights path')
    parser.add_argument('--project', type=str, default="runs/detect",
                        help='Project directory')
    parser.add_argument('--name', type=str, default="reefscape_yolov12",
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--skip-val', action='store_true',
                        help='Skip validation after training')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a specific model to use for validation and inference')
    parser.add_argument('--test-all', action='store_true',
                        help='Test on all test images instead of just a few random ones')
    parser.add_argument('--test-num', type=int, default=5,
                        help='Number of test images to select if not using --test-all')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for test image selection (default: uses time)')
    return parser.parse_args()

def verify_dataset(yaml_path):
    """Verify that dataset paths exist and are accessible."""
    yaml_path = Path(yaml_path)
    dataset_dir = yaml_path.parent
    
    # Load yaml file
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Check if paths exist
    train_path = dataset_dir / data['train']
    val_path = dataset_dir / data['val']
    test_path = dataset_dir / data['test']
    
    paths_ok = True
    
    if not train_path.exists():
        print(f"ERROR: Training images path not found: {train_path}")
        paths_ok = False
    else:
        print(f"Found {len(list(train_path.glob('*')))} training images")
    
    if not val_path.exists():
        print(f"ERROR: Validation images path not found: {val_path}")
        paths_ok = False
    else:
        print(f"Found {len(list(val_path.glob('*')))} validation images")
    
    if not test_path.exists():
        print(f"ERROR: Test images path not found: {test_path}")
        paths_ok = False
    else:
        print(f"Found {len(list(test_path.glob('*')))} test images")
    
    return paths_ok

def find_best_model(project_dir, name):
    """Find the best model in the project directory"""
    # Try exact path first
    model_dir = Path(project_dir) / name
    best_model_path = model_dir / "weights" / "best.pt"
    
    if best_model_path.exists():
        return best_model_path
    
    # If not found, try to find it in similar directories
    parent_dir = Path(project_dir)
    similar_dirs = list(parent_dir.glob(f"{name}*"))
    
    for d in similar_dirs:
        candidate = d / "weights" / "best.pt"
        if candidate.exists():
            print(f"Found model at: {candidate}")
            return candidate
    
    # If still not found, try last.pt
    last_model_path = model_dir / "weights" / "last.pt"
    if last_model_path.exists():
        print(f"Best model not found, using last.pt: {last_model_path}")
        return last_model_path
    
    # Try to find last.pt in similar directories
    for d in similar_dirs:
        candidate = d / "weights" / "last.pt"
        if candidate.exists():
            print(f"Found model at: {candidate}")
            return candidate
    
    print(f"WARNING: No model found in {project_dir}/{name} or similar directories")
    return None

def main():
    args = parse_args()
    
    # Setup paths
    dataset_path = Path(args.data)
    model_save_dir = Path(args.project) / args.name
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"MPS (Metal Performance Shaders) is available! Using Mac GPU acceleration.")
    else:
        device = "cpu"
        print(f"MPS is not available. Using CPU.")
    
    # Determine batch size
    if args.batch_size:
        batch_size = args.batch_size
    elif device == "mps":
        batch_size = 16  # Larger batch size for GPU
    else:
        batch_size = 8   # Smaller batch size for CPU
    
    # Verify dataset
    print("Verifying dataset paths...")
    if not verify_dataset(dataset_path):
        print("Dataset verification failed. Please check the paths in data.yaml")
        return
    
    # Training phase
    if not args.model_path:  # Skip training if model_path is provided
        # Check for resume
        resume = False
        if args.resume:
            resume_path = Path(args.resume)
            if resume_path.exists():
                print(f"Resuming training from checkpoint: {resume_path}")
                resume = True
                weights = resume_path
            else:
                print(f"Warning: Checkpoint file {resume_path} not found. Starting training from pretrained weights.")
                weights = args.weights
        else:
            weights = args.weights
        
        # 1. TRAINING
        print("=" * 80)
        print(f"{'Resuming' if resume else 'Starting'} YOLOv12 training on the 2025 Reefscape dataset")
        print(f"Using device: {device}, batch size: {batch_size}")
        print("=" * 80)
        
        # Load a YOLOv12 model
        model = YOLO(weights)
        
        # Train the model
        results = model.train(
            data=str(dataset_path.absolute()),  # Use absolute path
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch_size,
            device=device,
            patience=args.patience,  # Early stopping patience
            name=args.name,
            project=args.project,
            pretrained=True,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            close_mosaic=10,
            augment=True,
            cache=True,  # Cache images for faster training
            plots=True,
            save=True,
            save_period=-1,  # Save checkpoint every n epochs (-1 for last epoch only)
            resume=resume,  # Resume from checkpoint if specified
        )
        
        # Search for the best model file
        trained_model_path = find_best_model(args.project, args.name)
        if not trained_model_path:
            print("Error: Could not find trained model. Skipping validation and inference.")
            return
    else:
        # Use the provided model path
        trained_model_path = Path(args.model_path)
        if not trained_model_path.exists():
            print(f"Error: Provided model path does not exist: {trained_model_path}")
            return
        print(f"Using provided model: {trained_model_path}")
    
    # Skip validation if requested
    if args.skip_val:
        print("Skipping validation as requested.")
    else:
        # 2. VALIDATION
        print("\n" + "=" * 80)
        print("Validating YOLOv12 model on the 2025 Reefscape validation set")
        print("=" * 80)
        
        try:
            # Load the best model
            best_model = YOLO(trained_model_path)
            
            # Validate the model
            val_results = best_model.val(
                data=str(dataset_path.absolute()),  # Use absolute path
                imgsz=args.imgsz,
                batch=batch_size,
                device=device,
                plots=True,
                save_json=True,
                save_hybrid=True,
            )
            
            print(f"Validation metrics: mAP50 = {val_results.box.map50:.4f}, mAP50-95 = {val_results.box.map:.4f}")
        except Exception as e:
            print(f"Error during validation: {e}")
            print("Continuing to inference demo...")
            best_model = YOLO(trained_model_path)
    
    # 3. INFERENCE DEMO
    print("\n" + "=" * 80)
    print("Demonstrating inference on 2025 Reefscape test images")
    print("=" * 80)
    
    try:
        # Make sure best_model is defined
        if 'best_model' not in locals():
            best_model = YOLO(trained_model_path)
            
        # Get test image paths
        test_dir = Path("2025 REEFSCAPE.v2i.yolov11/test/images")
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
        
        if not test_images:
            print("No test images found!")
            return
        
        # Use a different random seed for test image selection
        # This ensures different images are selected each time
        if args.random_seed is not None:
            test_random_seed = args.random_seed
        else:
            test_random_seed = int(time.time())  # Use current time as seed
        
        print(f"Using random seed {test_random_seed} for test image selection")
        random.seed(test_random_seed)
        
        # Select test images
        if args.test_all:
            print(f"Testing on all {len(test_images)} test images")
            demo_images = test_images
        else:
            num_test = min(args.test_num, len(test_images))
            print(f"Randomly selecting {num_test} test images from {len(test_images)} available")
            demo_images = random.sample(test_images, num_test)
        
        # Run inference on these images
        for i, img_path in enumerate(demo_images):
            print(f"[{i+1}/{len(demo_images)}] Running inference on {img_path.name}")
            results = best_model.predict(
                source=img_path,
                conf=0.25,       # Confidence threshold
                iou=0.45,        # NMS IoU threshold
                max_det=300,     # Maximum detections per image
                save=True,       # Save results
                imgsz=args.imgsz, # Inference size
                visualize=False, # Visualize features
                augment=False,   # Augmented inference
                agnostic_nms=False, # Class-agnostic NMS
                verbose=False,   # Verbose output
                device=device,   # Use Metal GPU acceleration for Mac M-series
            )
            
            # Display results
            res = results[0]
            
            # Get the original image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw bounding boxes and labels
            annotated_img = res.plot()
            
            # Display image with detections
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_img)
            plt.title(f"YOLOv12 Detections on {img_path.name}")
            plt.axis("off")
            plt.tight_layout()
            
            # Save the visualized results
            inference_dir = Path(args.project) / "inference_results"
            inference_dir.mkdir(exist_ok=True, parents=True)
            save_path = inference_dir / f"demo_{i+1:02d}_{img_path.name}"
            plt.savefig(save_path)
            print(f"Result saved to {save_path}")
            
            # Close the figure to free memory
            plt.close()
        
        # Print model performance summary
        print("\nModel performance summary:")
        if 'val_results' in locals():
            print(f"mAP50: {val_results.box.map50:.4f}")
            print(f"mAP50-95: {val_results.box.map:.4f}")
            print(f"Precision: {val_results.box.p:.4f}")
            print(f"Recall: {val_results.box.r:.4f}")
    
    except Exception as e:
        print(f"Error during inference demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("YOLOV12 Reefscape model training and evaluation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main() 