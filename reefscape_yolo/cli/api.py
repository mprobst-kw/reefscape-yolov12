#!/usr/bin/env python3
"""REST API service for YOLOv12 inference on the REEFSCAPE dataset."""

import os
import io
import argparse
import base64
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
import torch
from flask import Flask, request, jsonify

from reefscape_yolo.core.model import load_model
from reefscape_yolo.core.inference import process_image
from reefscape_yolo.config.defaults import API_DEFAULTS

# Global variables
model = None
model_path = None
default_conf = API_DEFAULTS['conf']
default_iou = API_DEFAULTS['iou']
default_img_size = API_DEFAULTS['imgsz']
temp_dir = None

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": str(model_path),
        "device": model.device if hasattr(model, 'device') else "unknown"
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
    try:
        result = process_image(
            model=model,
            image_data=image_data,
            conf=conf,
            iou=iou,
            img_size=img_size,
            normalized_coords=normalized_coords
        )
        
        # Add timing information
        processing_time = time.time() - start_time
        result["timing"] = {
            "processing_time_seconds": processing_time
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    global model, model_path, temp_dir
    
    parser = argparse.ArgumentParser(description="Start a REST API service for YOLOv12 inference")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (e.g., best.pt or last.pt)')
    parser.add_argument('--device', type=str, default=API_DEFAULTS['device'],
                        help='Device to run on (e.g., "mps", "cpu")')
    parser.add_argument('--host', type=str, default=API_DEFAULTS['host'],
                        help=f'Host to run the server on (default: {API_DEFAULTS["host"]})')
    parser.add_argument('--port', type=int, default=API_DEFAULTS['port'],
                        help=f'Port to run the server on (default: {API_DEFAULTS["port"]})')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--conf', type=float, default=API_DEFAULTS['conf'],
                        help=f'Default confidence threshold (default: {API_DEFAULTS["conf"]})')
    parser.add_argument('--iou', type=float, default=API_DEFAULTS['iou'],
                        help=f'Default IoU threshold (default: {API_DEFAULTS["iou"]})')
    
    args = parser.parse_args()
    
    # Update default thresholds
    global default_conf, default_iou
    default_conf = args.conf
    default_iou = args.iou
    
    # Save parameters
    model_path = args.model
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create a temporary directory for uploaded images
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    # Load the model
    model = load_model(model_path, args.device)
    
    # Add some warmup inference to initialize the model fully
    print("Warming up model with a test inference...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Warmup inference
    process_image(model, dummy_img, default_conf, default_iou, default_img_size)
    print("Model warmup complete")
    
    # Run the server
    print(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main() 