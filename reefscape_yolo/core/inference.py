"""Inference utilities for REEFSCAPE YOLOv12 project."""

import os
import tempfile
from pathlib import Path
import numpy as np
from reefscape_yolo.core.model import load_model
from reefscape_yolo.utils.image import load_image, array_to_bytes


def process_image(model, image_data, conf=0.25, iou=0.45, img_size=640, normalized_coords=False):
    """Process an image and return detections.
    
    Args:
        model: YOLO model
        image_data: Image data (can be path, bytes, base64, or array)
        conf (float, optional): Confidence threshold. Defaults to 0.25.
        iou (float, optional): IOU threshold. Defaults to 0.45.
        img_size (int, optional): Image size. Defaults to 640.
        normalized_coords (bool, optional): Whether to return normalized coordinates. Defaults to False.
        
    Returns:
        dict: Dictionary with detections
    """
    # Handle different image input types
    if isinstance(image_data, (str, Path)) and Path(image_data).exists():
        # File path
        source = str(Path(image_data).absolute())
    elif isinstance(image_data, np.ndarray):
        # Numpy array
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            temp_img_path = temp_img.name
            temp_img.write(array_to_bytes(image_data))
        source = temp_img_path
    else:
        # Bytes or base64
        try:
            # Try to load as some kind of image
            img_array = load_image(image_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
                temp_img_path = temp_img.name
                temp_img.write(array_to_bytes(img_array))
            source = temp_img_path
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
    
    try:
        # Run inference
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=300,
            imgsz=img_size,
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
                class_name = results.names[class_id]
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
        output = {
            "num_detections": len(detections),
            "detections": detections,
            "metadata": {
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "image_size": img_size,
                "normalized_coordinates": normalized_coords
            }
        }
        
        return output
    
    except Exception as e:
        raise Exception(f"Error during inference: {e}")
    
    finally:
        # Clean up the temporary file if we created one
        if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
            try:
                os.unlink(temp_img_path)
            except:
                pass 