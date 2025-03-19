#!/usr/bin/env python3
# Client script to test the YOLOv12 REST API service

import argparse
import requests
import json
import base64
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Test the YOLOv12 REST API service")
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='The base URL of the API server (default: http://localhost:5000)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to test inference on')
    parser.add_argument('--use-path', action='store_true',
                        help='Use the image path instead of base64 encoding (only works if server and client are on the same machine)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--normalized', action='store_true',
                        help='Request normalized coordinates (0-1) instead of pixel values')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the JSON output (default: same name as input with .api.json extension)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.with_suffix('.api.json')
    
    # First check if the server is running
    try:
        health_url = f"{args.url}/health"
        response = requests.get(health_url)
        if response.status_code != 200:
            print(f"Error: Server health check failed with status code {response.status_code}")
            return
        
        health_data = response.json()
        print(f"Server is running with model: {health_data.get('model')}")
        print(f"Server device: {health_data.get('device')}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to server at {args.url}")
        print(f"Error details: {e}")
        return
    
    # Prepare the request
    predict_url = f"{args.url}/predict"
    
    if args.use_path:
        # Use the image path (only works if server and client are on the same machine)
        payload = {
            "image_path": str(image_path.absolute()),
            "conf": args.conf,
            "normalized_coords": args.normalized
        }
    else:
        # Read the image and encode as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        payload = {
            "image_base64": image_data,
            "conf": args.conf,
            "normalized_coords": args.normalized
        }
    
    # Send the request and measure time
    print(f"Sending request to {predict_url}...")
    start_time = time.time()
    
    try:
        response = requests.post(predict_url, json=payload)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Save the result to a file
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Print a summary
            num_detections = result.get("num_detections", 0)
            server_time = result.get("timing", {}).get("processing_time_seconds", 0)
            print(f"Success! Found {num_detections} objects in {total_time:.2f} seconds")
            print(f"Server processing time: {server_time:.2f} seconds")
            print(f"Network overhead: {total_time - server_time:.2f} seconds")
            
            # Print detections
            class_counts = {}
            for det in result.get("detections", []):
                cls_name = det["class_name"]
                confidence = det["confidence"]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                print(f"  {cls_name}: confidence={confidence:.2f}, bbox={det['bbox']}")
            
            print("\nClass summary:")
            for cls_name, count in class_counts.items():
                print(f"  {cls_name}: {count}")
            
            print(f"\nFull results saved to {output_path}")
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            print(response.text)
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    main() 