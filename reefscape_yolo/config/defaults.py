"""Default configuration values for the REEFSCAPE YOLOv12 project."""

import os
from pathlib import Path
from reefscape_yolo.core.dataset import find_dataset_yaml

# Find dataset path
dataset_yaml = find_dataset_yaml()
DEFAULT_DATASET = str(dataset_yaml) if dataset_yaml else 'data/reefscape/data.yaml'

# Default training parameters
TRAINING_DEFAULTS = {
    'epochs': 50,
    'patience': 10,
    'imgsz': 640,
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'cos_lr': True,
    'close_mosaic': 10,
    'augment': True,
    'cache': True,
    'project': 'runs/detect',
    'name': 'reefscape_yolov12',
    'pretrained': True,
    'batch_size_gpu': 16,
    'batch_size_cpu': 8,
    'seed': 42,
}

# Default inference parameters
INFERENCE_DEFAULTS = {
    'conf': 0.25,
    'iou': 0.45,
    'imgsz': 640,
    'max_det': 300,
    'device': None,
    'save_txt': False,
    'save_conf': False,
    'save_json': False,
    'save_img': True,
    'project': 'runs/detect',
    'name': 'exp',
}

# Default validation parameters
VALIDATION_DEFAULTS = {
    'conf': 0.25,
    'iou': 0.45,
    'imgsz': 640,
    'batch_size': 8,
    'device': None,
    'save_json': False,
    'save_hybrid': False,
    'plots': True,
    'project': 'validation_results',
    'name': 'exp',
}

# Default API parameters
API_DEFAULTS = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'conf': 0.25,
    'iou': 0.45,
    'imgsz': 640,
    'device': None,
}

# Model size variants and their corresponding weights
MODEL_VARIANTS = {
    'nano': 'yolov12n.pt',
    'small': 'yolov12s.pt',
    'medium': 'yolov12m.pt',
    'large': 'yolov12l.pt',
    'extra-large': 'yolov12x.pt',
}

# For versions with 'yolov12/' submodule, prepend the path
if Path('yolov12').is_dir():
    for key, value in MODEL_VARIANTS.items():
        if not Path(value).exists() and Path(f'yolov12/{value}').exists():
            MODEL_VARIANTS[key] = f'yolov12/{value}'

# Default submodule path
YOLOV12_SUBMODULE = 'yolov12' if Path('yolov12').is_dir() else None 