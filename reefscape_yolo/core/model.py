"""Model utilities for REEFSCAPE YOLOv12 project."""

import os
from pathlib import Path
import torch
from ultralytics import YOLO


def get_optimal_device():
    """Determine the optimal device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(model_path, device=None):
    """Load the YOLOv12 model.
    
    Args:
        model_path (str): Path to the model weights
        device (str, optional): Device to load the model on. 
            If None, automatically selects the best available device.
            
    Returns:
        YOLO: Loaded model
    """
    if device is None:
        device = get_optimal_device()
        
    if device == "mps":
        print("Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
    elif device == "cuda":
        print(f"Using CUDA for GPU acceleration")
    else:
        print(f"Using CPU for inference")
    
    # Load the model
    model = YOLO(model_path)
    return model


def find_best_model(project_dir, name):
    """Find the best model in the project directory.
    
    Args:
        project_dir (str): Project directory containing runs
        name (str): Experiment name
        
    Returns:
        Path: Path to the best model if found, else None
    """
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