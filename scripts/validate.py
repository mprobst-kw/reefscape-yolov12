#!/usr/bin/env python3
# Validation script for YOLOv12 model on the 2025 Reefscape dataset

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv12 model on the Reefscape dataset")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (e.g., best.pt or last.pt)')
    parser.add_argument('--data', type=str, default="2025 REEFSCAPE.v2i.yolov11/data.yaml",
                        help='Path to dataset configuration file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run validation on (e.g., "mps", "cpu")')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for validation')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Size of input images')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='Save hybrid version with labels')
    parser.add_argument('--plots', action='store_true',
                        help='Generate validation plots')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    return parser.parse_args()

def plot_pr_curve(val_results, save_dir='validation_results'):
    """Plot precision-recall curve"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Get mean precision and recall if they're arrays
    p = val_results.box.p
    r = val_results.box.r
    
    if isinstance(p, np.ndarray) and len(p) > 0:
        p = p.mean() if p.size > 1 else float(p[0])
    if isinstance(r, np.ndarray) and len(r) > 0:
        r = r.mean() if r.size > 1 else float(r[0])
    
    # Precision-Recall Curve
    plt.figure(figsize=(9, 6))
    plt.scatter(r, p, marker='o', color='blue', s=100)
    plt.grid(alpha=0.5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall (mAP={val_results.box.map:.4f})')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(save_dir / 'precision_recall_curve.png')
    plt.close()

def plot_f1_curve(val_results, save_dir='validation_results'):
    """Plot F1-confidence curve"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    if hasattr(val_results.box, 'f1') and hasattr(val_results.box, 'conf'):
        f1 = val_results.box.f1
        conf = val_results.box.conf
        
        if isinstance(f1, np.ndarray) and isinstance(conf, np.ndarray):
            if len(f1) > 0 and len(conf) > 0:
                # F1-Confidence Curve
                plt.figure(figsize=(9, 6))
                plt.scatter(conf, f1, marker='o', color='green', s=100)
                plt.grid(alpha=0.5)
                plt.xlabel('Confidence')
                plt.ylabel('F1 Score')
                plt.title('F1-Confidence')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.savefig(save_dir / 'f1_confidence_curve.png')
                plt.close()

def ensure_scalar(value):
    """Convert numpy arrays or tensors to scalar float values"""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if hasattr(value, 'size') and value.size > 0:
            return float(value.mean()) if value.size > 1 else float(value.item() if hasattr(value, 'item') else value[0])
    return float(value)

def main():
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
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
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Load dataset info
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"Error: Dataset configuration file not found: {dataset_path}")
        return
    
    # Validate the model
    print(f"\nValidating on {dataset_path}...")
    print(f"Using device: {device}, batch size: {batch_size}, image size: {args.imgsz}")
    
    val_results = model.val(
        data=str(dataset_path.absolute()),
        imgsz=args.imgsz,
        batch=batch_size,
        device=device,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        plots=args.plots,
        save_txt=args.save_txt,
        verbose=args.verbose
    )
    
    # Print validation metrics in a formatted table
    print("\n" + "=" * 80)
    print(" " * 30 + "VALIDATION RESULTS")
    print("=" * 80)
    
    # Main metrics - ensure scalar values
    try:
        metrics = [
            ("mAP50", ensure_scalar(val_results.box.map50)),
            ("mAP50-95", ensure_scalar(val_results.box.map)),
        ]
        
        # Add precision if available
        if hasattr(val_results.box, 'p'):
            metrics.append(("Precision", ensure_scalar(val_results.box.p)))
            
        # Add recall if available
        if hasattr(val_results.box, 'r'):
            metrics.append(("Recall", ensure_scalar(val_results.box.r)))
        
        # Add F1 score if available
        if hasattr(val_results.box, 'f1'):
            metrics.append(("F1-score", ensure_scalar(val_results.box.f1)))
        
        max_name_len = max(len(name) for name, _ in metrics)
        for name, value in metrics:
            print(f"{name:{max_name_len}} : {value:.6f}")
        
        # Class metrics if available
        if hasattr(val_results, 'names') and hasattr(val_results.box, 'ap_class_index'):
            print("\nPer-class metrics:")
            for i, class_idx in enumerate(val_results.box.ap_class_index):
                if class_idx < len(val_results.names):
                    class_name = val_results.names[class_idx]
                    print(f"{class_name:15} : mAP50 = {ensure_scalar(val_results.box.ap50[i]):.6f}")
        
        # Plot PR curve and F1 curve
        if args.plots:
            try:
                plot_pr_curve(val_results)
                plot_f1_curve(val_results)
                print("\nPlots saved to validation_results/ directory")
            except Exception as e:
                print(f"Warning: Could not create plots: {e}")
    
    except Exception as e:
        print(f"Error displaying metrics: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main() 