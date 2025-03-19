"""Dataset utilities for REEFSCAPE YOLOv12 project."""

import os
from pathlib import Path
import yaml
import shutil


def find_dataset_yaml():
    """Find the REEFSCAPE dataset YAML file in standard locations.
    
    Returns:
        Path: Path to the dataset YAML file if found, else None
    """
    # Standard locations to check
    possible_paths = [
        Path("data/reefscape/data.yaml"),
        Path("data/2025 REEFSCAPE.v2i.yolov11/data.yaml"),
        Path("data/2025 REEFSCAPE/data.yaml"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Search in data directory if not found in standard locations
    data_dir = Path("data")
    if data_dir.exists():
        yaml_files = list(data_dir.glob("**/data.yaml"))
        if yaml_files:
            return yaml_files[0]
    
    return None


def verify_dataset(yaml_path):
    """Verify that dataset paths exist and are accessible.
    
    Args:
        yaml_path (str or Path): Path to the dataset YAML file
        
    Returns:
        bool: True if all paths exist, False otherwise
    """
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


def get_dataset_info(yaml_path):
    """Get information about the dataset from the YAML file.
    
    Args:
        yaml_path (str or Path): Path to the dataset YAML file
        
    Returns:
        dict: Dataset information
    """
    yaml_path = Path(yaml_path)
    
    # Load yaml file
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract information
    num_classes = len(data.get('names', []))
    class_names = data.get('names', [])
    
    return {
        'num_classes': num_classes,
        'class_names': class_names,
        'yaml_path': str(yaml_path.absolute()),
        'train_path': data.get('train', ''),
        'val_path': data.get('val', ''),
        'test_path': data.get('test', ''),
    }


def setup_dataset_symlink():
    """Create a symbolic link from the actual dataset location to the standard location.
    
    This allows scripts to use the standard path (data/reefscape) even if the
    dataset is stored in a different location.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Find the actual dataset location
    yaml_path = find_dataset_yaml()
    if not yaml_path:
        print("ERROR: Could not find dataset YAML file.")
        return False
    
    # Standard location
    standard_dir = Path("data/reefscape")
    
    # If standard location already exists and is a directory (not a symlink),
    # don't overwrite it
    if standard_dir.exists() and not standard_dir.is_symlink():
        print(f"WARNING: {standard_dir} already exists and is not a symlink.")
        return False
    
    # Remove existing symlink if it exists
    if standard_dir.is_symlink():
        standard_dir.unlink()
    
    # Create parent directory if it doesn't exist
    standard_dir.parent.mkdir(exist_ok=True)
    
    # Create symlink
    dataset_dir = yaml_path.parent
    os.symlink(dataset_dir, standard_dir, target_is_directory=True)
    print(f"Created symlink: {dataset_dir} -> {standard_dir}")
    
    return True 