#!/usr/bin/env python3
# REST API service for YOLOv12 inference on Orange Pi 5

import os
import io
import json
import argparse
import base64
from pathlib import Path
import tempfile
from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Global variables
model = None
device = None
model_path = None
default_conf = 0.25
default_iou = 0.45
default_img_size = 640
temp_dir = None

app = Flask(__name__)

def load_model(model_path, device=None):
    """Load the YOLOv12 model"""
    global model
    
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    
    # Determine device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"MPS (Metal Performance Shaders) is available! Using Mac GPU acceleration.")
        else:
            device = "cpu"
            print(f"MPS is not available. Using CPU.")
    
    # Load the model
    model = YOLO(model_path)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    return model

def process_image(image_data, conf, iou, img_size, normalized_coords=False):
    """Process an image and return detections"""
    global model, device, temp_dir
    
    # Create a temporary file for the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=temp_dir) as temp_img:
        temp_img.write(image_data)
        temp_img_path = temp_img.name
    
    try:
        # Run inference
        results = model.predict(
            source=temp_img_path,
            conf=conf,
            iou=iou,
            max_det=300,
            imgsz=img_size,
            device=device,
            verbose=False
        )[0]  # Get the first (and only) result
        
        # Extract detections
        detections = []
        
        # Get image dimensions for normalization if needed
        if normalized_coords and hasattr(results, 'orig_shape'):
            img_height, img_width = results.orig_shape
        else:
            img_height, img_width = None, None
        
        # Process detection boxes
        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Normalize coordinates if requested
                if normalized_coords and img_height and img_width:
                    x1 /= img_width
                    x2 /= img_width
                    y1 /= img_height
                    y2 /= img_height
                
                # Get class info
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = float(box.conf[0].item())
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                }
                
                detections.append(detection)
        
        # Create output JSON
        output_json = {
            "num_detections": len(detections),
            "detections": detections,
            "metadata": {
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "image_size": img_size,
                "normalized_coordinates": normalized_coords
            }
        }
        
        return output_json, 200
    
    except Exception as e:
        return {"error": str(e)}, 500
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_img_path)
        except:
            pass

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": str(model_path),
        "device": device
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for prediction"""
    start_time = time.time()
    
    # Get parameters from request
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get image data - either from a URL, a base64 string, or a file path
    if 'image_base64' in request.json:
        try:
            image_data = base64.b64decode(request.json['image_base64'])
        except:
            return jsonify({"error": "Invalid base64 image data"}), 400
    elif 'image_path' in request.json:
        try:
            image_path = request.json['image_path']
            with open(image_path, 'rb') as f:
                image_data = f.read()
        except:
            return jsonify({"error": f"Cannot read image from path: {image_path}"}), 400
    else:
        return jsonify({"error": "No image provided. Use 'image_base64' or 'image_path'"}), 400
    
    # Get inference parameters
    conf = request.json.get('conf', default_conf)
    iou = request.json.get('iou', default_iou)
    img_size = request.json.get('img_size', default_img_size)
    normalized_coords = request.json.get('normalized_coords', False)
    
    # Process the image
    result, status_code = process_image(image_data, conf, iou, img_size, normalized_coords)
    
    # Add timing information
    processing_time = time.time() - start_time
    if status_code == 200:
        result["timing"] = {
            "processing_time_seconds": processing_time
        }
    
    return jsonify(result), status_code

def main():
    global model_path, device, temp_dir
    
    parser = argparse.ArgumentParser(description="Start a REST API service for YOLOv12 inference")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (e.g., best.pt or last.pt)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Save parameters
    model_path = args.model
    device = args.device
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create a temporary directory for uploaded images
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    # Load the model
    load_model(model_path, device)
    
    # Add some warmup inference to initialize the model fully
    print("Warming up model with a test inference...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    pil_img = Image.fromarray(dummy_img)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Warmup inference
    process_image(img_byte_arr, default_conf, default_iou, default_img_size)
    print("Model warmup complete")
    
    # Run the server
    print(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == "__main__":
    main() 