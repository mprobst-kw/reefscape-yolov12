#!/usr/bin/env python3
"""
Setup script to initialize the REEFSCAPE YOLOv12 project.

This script:
1. Checks for the YOLOv12 submodule and initializes it if needed
2. Finds the REEFSCAPE dataset and creates a symlink in the standard location
3. Installs the package in development mode
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, cwd=None):
    """Run a shell command and return the output."""
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None


def setup_yolov12_submodule():
    """Check for YOLOv12 submodule and initialize it if needed."""
    yolov12_dir = Path("yolov12")
    
    if yolov12_dir.exists() and len(list(yolov12_dir.glob("*"))) > 0:
        print("YOLOv12 submodule already exists.")
        return True
    
    print("Initializing YOLOv12 submodule...")
    
    # Check if .gitmodules exists
    if Path(".gitmodules").exists():
        # Update existing submodule
        run_command(["git", "submodule", "update", "--init", "--recursive"])
    else:
        # Add submodule
        run_command(["git", "submodule", "add", "https://github.com/ultralytics/yolov12.git"])
        run_command(["git", "submodule", "update", "--init", "--recursive"])
    
    # Verify it worked
    if yolov12_dir.exists() and len(list(yolov12_dir.glob("*"))) > 0:
        print("YOLOv12 submodule initialized successfully.")
        return True
    else:
        print("ERROR: Failed to initialize YOLOv12 submodule.")
        print("Please manually clone the YOLOv12 repository.")
        return False


def setup_dataset_symlink():
    """Find the REEFSCAPE dataset and create a symlink in the standard location."""
    from reefscape_yolo.core.dataset import setup_dataset_symlink, find_dataset_yaml
    
    dataset_yaml = find_dataset_yaml()
    if dataset_yaml:
        print(f"Found dataset at: {dataset_yaml}")
        if setup_dataset_symlink():
            print("Dataset symlink created successfully.")
            return True
        else:
            print("Failed to create dataset symlink.")
            return False
    else:
        print("ERROR: Could not find REEFSCAPE dataset.")
        print("Please place the dataset in one of the following locations:")
        print("  - data/reefscape/")
        print("  - data/2025 REEFSCAPE.v2i.yolov11/")
        return False


def install_package():
    """Install the package in development mode."""
    print("Installing package in development mode...")
    result = run_command([sys.executable, "-m", "pip", "install", "-e", "."])
    
    if result is not None:
        print("Package installed successfully.")
        return True
    else:
        print("ERROR: Failed to install package.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup REEFSCAPE YOLOv12 project")
    parser.add_argument('--skip-submodule', action='store_true',
                        help='Skip YOLOv12 submodule setup')
    parser.add_argument('--skip-dataset', action='store_true',
                        help='Skip dataset symlink setup')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip package installation')
    args = parser.parse_args()
    
    # Create required directories
    for dir_path in ["data", "models", "runs", "validation_results"]:
        Path(dir_path).mkdir(exist_ok=True)
    
    # Setup YOLOv12 submodule
    if not args.skip_submodule:
        setup_yolov12_submodule()
    
    # Try importing the package
    try:
        import reefscape_yolo
    except ImportError:
        print("Installing package first so we can use its modules...")
        install_package()
    
    # Setup dataset symlink
    if not args.skip_dataset:
        setup_dataset_symlink()
    
    # Install package
    if not args.skip_install:
        install_package()
    
    print("\nSetup complete!")
    print("You can now use the REEFSCAPE YOLOv12 package.")
    print("\nExample commands:")
    print("  reefscape-train --model-size nano --epochs 50")
    print("  reefscape-validate --weights models/best.pt")
    print("  reefscape-predict --source data/reefscape/test --weights models/best.pt")
    print("  reefscape-api --model models/best.pt")


if __name__ == "__main__":
    main() 