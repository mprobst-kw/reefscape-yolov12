"""Image processing utilities for the REEFSCAPE YOLOv12 project."""

import io
import base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def load_image(image_source):
    """Load an image from different sources (path, bytes, base64).
    
    Args:
        image_source: Can be a path (str), bytes, base64 string, or numpy array
        
    Returns:
        numpy.ndarray: Image as a numpy array in RGB format
    """
    if isinstance(image_source, np.ndarray):
        # Already a numpy array
        img_array = image_source
    elif isinstance(image_source, (str, Path)):
        # Could be a file path or base64 string
        path = Path(image_source)
        if path.exists():
            # It's a file path
            img_array = cv2.imread(str(path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            # Try to decode as base64
            try:
                img_bytes = base64.b64decode(image_source)
                img_array = bytes_to_array(img_bytes)
            except Exception as e:
                raise ValueError(f"Invalid image source: {e}")
    elif isinstance(image_source, bytes):
        # Raw bytes
        img_array = bytes_to_array(image_source)
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    return img_array


def bytes_to_array(img_bytes):
    """Convert image bytes to numpy array.
    
    Args:
        img_bytes (bytes): Image as bytes
        
    Returns:
        numpy.ndarray: Image as a numpy array in RGB format
    """
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def array_to_bytes(img_array, format='JPEG'):
    """Convert numpy array to image bytes.
    
    Args:
        img_array (numpy.ndarray): Image as numpy array
        format (str, optional): Image format. Defaults to 'JPEG'.
        
    Returns:
        bytes: Image as bytes
    """
    img = Image.fromarray(img_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on an image.
    
    Args:
        image (numpy.ndarray): Image to draw on
        detections (list): List of detection dictionaries
        color (tuple, optional): Box color. Defaults to (0, 255, 0).
        thickness (int, optional): Line thickness. Defaults to 2.
        
    Returns:
        numpy.ndarray: Image with detections drawn
    """
    # Make a copy of the image
    img_draw = image.copy()
    
    # Draw each detection
    for det in detections:
        bbox = det['bbox']
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])
        
        # Draw the bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Draw the label
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(img_draw, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_draw 