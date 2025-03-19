#!/usr/bin/env python3
"""Train a YOLOv12 model on the REEFSCAPE dataset."""

import os
import random
import argparse
import time
import sys
from pathlib import Path
import numpy as np
import torch

from reefscape_yolo.core.model import load_model, find_best_model
from reefscape_yolo.core.dataset import verify_dataset, find_dataset_yaml, setup_dataset_symlink
from reefscape_yolo.config.defaults import DEFAULT_DATASET, TRAINING_DEFAULTS, MODEL_VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv12 on the REEFSCAPE dataset")
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a specific checkpoint file (e.g., path to best.pt or last.pt)')
    parser.add_argument('--epochs', type=int, default=TRAINING_DEFAULTS['epochs'],
                        help=f'Number of training epochs (default: {TRAINING_DEFAULTS["epochs"]})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training (default: auto-selected based on device)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--data', type=str, default=DEFAULT_DATASET,
                        help=f'Path to dataset configuration file (default: {DEFAULT_DATASET})')
    parser.add_argument('--imgsz', type=int, default=TRAINING_DEFAULTS['imgsz'],
                        help=f'Size of input images as integer (default: {TRAINING_DEFAULTS["imgsz"]})')
    parser.add_argument('--weights', type=str, default=MODEL_VARIANTS['nano'],
                        help=f'Initial weights path (default: {MODEL_VARIANTS["nano"]})')
    parser.add_argument('--project', type=str, default=TRAINING_DEFAULTS['project'],
                        help=f'Project directory (default: {TRAINING_DEFAULTS["project"]})')
    parser.add_argument('--name', type=str, default=TRAINING_DEFAULTS['name'],
                        help=f'Experiment name (default: {TRAINING_DEFAULTS["name"]})')
    parser.add_argument('--patience', type=int, default=TRAINING_DEFAULTS['patience'],
                        help=f'Early stopping patience (epochs) (default: {TRAINING_DEFAULTS["patience"]})')
    parser.add_argument('--skip-val', action='store_true',
                        help='Skip validation after training')
    parser.add_argument('--seed', type=int, default=TRAINING_DEFAULTS['seed'],
                        help=f'Random seed for reproducibility (default: {TRAINING_DEFAULTS["seed"]})')
    parser.add_argument('--model-size', type=str, choices=list(MODEL_VARIANTS.keys()), 
                        help='YOLOv12 model size (nano, small, medium, large, extra-large)')
    parser.add_argument('--setup-dataset', action='store_true',
                        help='Setup dataset symlink to standard location')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup dataset symlink if requested
    if args.setup_dataset:
        setup_dataset_symlink()
    
    # If model size is specified, use the corresponding weights
    if args.model_size and not args.weights:
        args.weights = MODEL_VARIANTS[args.model_size]
        print(f"Using {args.model_size} model: {args.weights}")
    
    # Set seed for reproducibility in training
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    model_save_dir = Path(args.project) / args.name
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Determine device
    device = args.device
    
    # Determine batch size
    if args.batch_size:
        batch_size = args.batch_size
    elif device == "mps" or device == "cuda":
        batch_size = TRAINING_DEFAULTS['batch_size_gpu']
    else:
        batch_size = TRAINING_DEFAULTS['batch_size_cpu']
    
    # Verify dataset
    print("Verifying dataset paths...")
    if not verify_dataset(dataset_path):
        print("Dataset verification failed. Please check the paths in data.yaml")
        return
    
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
    print(f"{'Resuming' if resume else 'Starting'} YOLOv12 training on the REEFSCAPE dataset")
    print(f"Using device: {device}, batch size: {batch_size}")
    print("=" * 80)
    
    # Load a YOLOv12 model
    model = load_model(weights, device)
    
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
        optimizer=TRAINING_DEFAULTS['optimizer'],
        lr0=TRAINING_DEFAULTS['lr0'],
        lrf=TRAINING_DEFAULTS['lrf'],
        momentum=TRAINING_DEFAULTS['momentum'],
        weight_decay=TRAINING_DEFAULTS['weight_decay'],
        warmup_epochs=TRAINING_DEFAULTS['warmup_epochs'],
        cos_lr=TRAINING_DEFAULTS['cos_lr'],
        close_mosaic=TRAINING_DEFAULTS['close_mosaic'],
        augment=TRAINING_DEFAULTS['augment'],
        cache=TRAINING_DEFAULTS['cache'],  # Cache images for faster training
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
    
    # Skip validation if requested
    if args.skip_val:
        print("Skipping validation as requested.")
    else:
        # 2. VALIDATION
        print("\n" + "=" * 80)
        print("Validating YOLOv12 model on the REEFSCAPE validation set")
        print("=" * 80)
        
        try:
            # Load the best model
            best_model = load_model(trained_model_path, device)
            
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
            
            print(f"Validation complete. Best model saved at: {trained_model_path}")
            print(f"Validation mAP@0.5: {val_results.box.map50:.4f}")
            print(f"Validation mAP@0.5:0.95: {val_results.box.map:.4f}")
        
        except Exception as e:
            print(f"Error during validation: {e}")
    
    print("\nTraining and validation complete!")
    print(f"Best model saved at: {trained_model_path}")
    

if __name__ == "__main__":
    main() 