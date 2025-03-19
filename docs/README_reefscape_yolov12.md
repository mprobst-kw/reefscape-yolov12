# YOLOv12 for 2025 Reefscape Dataset

This project demonstrates how to train, test, and deploy a YOLOv12 model for coral reef image analysis using the 2025 Reefscape dataset. The model detects and localizes two classes: ALGAE and CORAL.

## Project Structure

- `train_reefscape_yolov12.py`: Script for training and testing the YOLOv12 model
- `predict_reefscape_yolov12.py`: Script for running inference on new images using a trained model
- `2025 REEFSCAPE.v2i.yolov11/`: The Reefscape dataset

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended), ensure you have CUDA installed.

## Dataset

The 2025 Reefscape dataset contains images of coral reefs with annotations for two classes:
1. ALGAE
2. CORAL

The dataset is organized as follows:
- `train/`: Training images and labels
- `valid/`: Validation images and labels
- `test/`: Test images and labels
- `data.yaml`: Dataset configuration file

## Training

To train the YOLOv12 model on the Reefscape dataset, run:

```bash
python train_reefscape_yolov12.py
```

This script performs the following steps:
1. Loads a pre-trained YOLOv12 model
2. Trains the model on the Reefscape training dataset
3. Validates the trained model on the validation set
4. Runs inference on test images
5. Visualizes model performance

Training results, including checkpoints and performance metrics, are saved to `runs/detect/reefscape_yolov12/`.

### Training Parameters

The training script uses the following parameters:
- **Model**: YOLOv12n (nano version, smallest and fastest)
- **Epochs**: 50
- **Image Size**: 640×640
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Optimizer**: SGD with momentum 0.937
- **Early Stopping Patience**: 10 epochs

## Inference

To run inference on new images using the trained model:

```bash
python predict_reefscape_yolov12.py --source path/to/image_or_directory
```

### Additional Arguments

- `--model`: Path to the trained model file (default: `runs/detect/reefscape_yolov12/weights/best.pt`)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to run inference on ("0" for GPU, "cpu" for CPU)
- `--save-dir`: Directory to save results (default: `runs/detect/predict_results`)
- `--view-img`: Display results (add this flag to show images)

## Results

The model outputs:
- Bounding box coordinates for detected objects
- Class predictions (ALGAE or CORAL)
- Confidence scores

Results are saved in the following formats:
- Annotated images
- Text files with detection coordinates
- Cropped detections
- Summary statistics

## Performance Evaluation

The model is evaluated using standard object detection metrics:
- Precision
- Recall
- mAP50 (mean Average Precision at IoU threshold 0.5)
- mAP50-95 (mean Average Precision across IoU thresholds 0.5 to 0.95)

## Example Usage

Train the model:
```bash
python train_reefscape_yolov12.py
```

Run inference on a single image:
```bash
python predict_reefscape_yolov12.py --source path/to/image.jpg --view-img
```

Run inference on multiple images:
```bash
python predict_reefscape_yolov12.py --source path/to/images/ --conf 0.3
```
``` 