# REEFSCAPE YOLOv12 Project

This repository contains code for training, validating, and deploying YOLOv12 models on the REEFSCAPE dataset.

## Project Structure

```
reefscape-yolov12/                  # Root directory 
│
├── reefscape_yolo/                 # Main Python package
│   ├── core/                       # Core functionality
│   │   ├── model.py                # Model loading and management
│   │   ├── dataset.py              # Dataset utilities
│   │   └── inference.py            # Inference functionality
│   │
│   ├── utils/                      # Utility functions
│   │   └── image.py                # Image processing utilities
│   │
│   ├── cli/                        # Command-line interfaces
│   │   ├── train.py                # Training CLI
│   │   ├── validate.py             # Validation CLI
│   │   ├── predict.py              # Prediction CLI
│   │   └── api.py                  # REST API CLI
│   │
│   └── config/                     # Configuration
│       └── defaults.py             # Default configuration values
│
├── yolov12/                        # YOLOv12 submodule
│
├── data/                           # Dataset directory (not version controlled)
│   └── 2025 REEFSCAPE.v2i.yolov11/ # Actual dataset location
│
├── docs/                           # Documentation
│   ├── ORANGEPI_SETUP.md           # Instructions for Orange Pi setup
│   └── README_reefscape_yolov12.md # Original project README
│
├── models/                         # Trained model weights
│
├── runs/                           # Training runs output
│
├── validation_results/             # Validation results
│
├── examples/                       # Example scripts
│
├── setup_project.py                # Project setup script
├── setup.py                        # Package setup
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Installation

### Quick Setup

The easiest way to set up the project is to use the provided setup script:

```bash
# Clone the repository
git clone https://github.com/your-username/reefscape-yolov12.git
cd reefscape-yolov12

# Run the setup script
python setup_project.py
```

This script will:
1. Initialize the YOLOv12 submodule
2. Find the REEFSCAPE dataset and create a symlink
3. Install the package in development mode

### Manual Setup

If you prefer to set up the project manually:

#### 1. Clone this repository

```bash
git clone https://github.com/your-username/reefscape-yolov12.git
cd reefscape-yolov12
```

#### 2. Initialize and update the YOLOv12 submodule

```bash
git submodule update --init --recursive
```

#### 3. Set up a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Install the package in development mode

```bash
pip install -e .
```

#### 5. Set up the REEFSCAPE dataset

The REEFSCAPE dataset should be located in one of the following paths:
- `data/reefscape/`
- `data/2025 REEFSCAPE.v2i.yolov11/`

If it's in a different location, you can use the dataset utilities to create a symbolic link:

```python
from reefscape_yolo.core.dataset import setup_dataset_symlink
setup_dataset_symlink()
```

## Usage

### Training

```bash
# Using the CLI script directly
python -m reefscape_yolo.cli.train --data data/2025\ REEFSCAPE.v2i.yolov11/data.yaml --epochs 50

# Using the installed command
reefscape-train --epochs 50 --model-size nano

# Using a specific model size
reefscape-train --model-size small --epochs 100
```

### Validation

```bash
# Using the CLI script directly
python -m reefscape_yolo.cli.validate --weights models/best.pt

# Using the installed command
reefscape-validate --weights models/best.pt
```

### Prediction

```bash
# Using the CLI script directly
python -m reefscape_yolo.cli.predict --source data/2025\ REEFSCAPE.v2i.yolov11/test/images --weights models/best.pt

# Using the installed command
reefscape-predict --source test --weights models/best.pt

# Specifying output options
reefscape-predict --source test --weights models/best.pt --save-json --save-img
```

### API Usage

```bash
# Using the CLI script directly
python -m reefscape_yolo.cli.api --model models/best.pt

# Using the installed command
reefscape-api --model models/best.pt --port 8000
```

## API Endpoints

The API offers the following endpoints:

### Health Check

```
GET /health
```

Returns the status of the API and model information.

### Prediction

```
POST /predict
```

Body parameters:
- `image_base64`: Base64-encoded image data, or
- `image_path`: Path to image file
- `conf` (optional): Confidence threshold (default: 0.25)
- `iou` (optional): IoU threshold (default: 0.45)
- `img_size` (optional): Image size (default: 640)
- `normalized_coords` (optional): Return normalized coordinates (default: false)

## Using as a Library

You can also use the package as a library in your own Python code:

```python
from reefscape_yolo.core.model import load_model
from reefscape_yolo.core.inference import process_image

# Load a model
model = load_model("models/best.pt")

# Process an image
results = process_image(model, "path/to/image.jpg")

# Print detections
for det in results["detections"]:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

For more examples, see the `examples/` directory.

## Keeping YOLOv12 Updated

To update the YOLOv12 submodule to the latest version:

```bash
cd yolov12
git pull origin main
cd ..
git add yolov12
git commit -m "Update YOLOv12 submodule"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. Note that the YOLOv12 repository has its own license.

## Acknowledgements

- [Ultralytics YOLOv12](https://github.com/ultralytics/yolov12)
- REEFSCAPE dataset contributors 