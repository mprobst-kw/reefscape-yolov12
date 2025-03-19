# YOLOv12 REST API Service for Orange Pi 5

This guide explains how to set up and run the YOLOv12 REST API service on an Orange Pi 5, optimizing for inference latency by keeping the model loaded in memory.

## Hardware Requirements

- Orange Pi 5 with at least 4GB RAM (8GB recommended)
- microSD card (32GB+ recommended)
- Power supply compatible with Orange Pi 5

## Software Setup

### 1. Install the Operating System

If you haven't already, install Ubuntu 22.04 or Armbian on your Orange Pi 5:

```bash
# Download Ubuntu image from the official Orange Pi website or use Armbian
# Flash the image to your microSD card using tools like balenaEtcher
```

### 2. Update System and Install Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git wget libopenblas-dev libjpeg-dev zlib1g-dev
```

### 3. Create a Python Virtual Environment

```bash
mkdir -p ~/yolov12_service
cd ~/yolov12_service
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Required Python Packages

```bash
# Install PyTorch for ARM
pip install --upgrade pip
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### 5. Download Your Trained YOLOv12 Model

Transfer your trained model to the Orange Pi:

```bash
mkdir -p ~/yolov12_service/models
# Transfer your trained model (e.g., best.pt) to this directory
# You can use scp, rsync, or a USB drive for this
```

## Running the Service

### Start the API Service

```bash
cd ~/yolov12_service
source venv/bin/activate

# Run the API server
python reefscape_yolo_api.py --model models/best.pt --port 5000
```

You should see output showing that the model is loading and warming up, followed by a message that the server is running.

### Running as a System Service for Auto-start

To ensure the service starts automatically when the Orange Pi boots, create a systemd service:

```bash
sudo nano /etc/systemd/system/yolov12-api.service
```

Add the following content:

```
[Unit]
Description=YOLOv12 REST API Service
After=network.target

[Service]
User=orangepi
WorkingDirectory=/home/orangepi/yolov12_service
ExecStart=/home/orangepi/yolov12_service/venv/bin/python /home/orangepi/yolov12_service/reefscape_yolo_api.py --model /home/orangepi/yolov12_service/models/best.pt --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace `orangepi` with your username.

Then enable and start the service:

```bash
sudo systemctl enable yolov12-api
sudo systemctl start yolov12-api
sudo systemctl status yolov12-api
```

## Testing the Service

### From the Orange Pi itself

```bash
cd ~/yolov12_service
source venv/bin/activate

# Test using a local image
python test_yolo_api.py --image path/to/test_image.jpg --use-path
```

### From another device on the network

```bash
# Replace 192.168.1.100 with your Orange Pi's IP address
python test_yolo_api.py --url http://192.168.1.100:5000 --image path/to/test_image.jpg
```

## Performance Optimization

### 1. Model Quantization

To reduce model size and improve inference speed:

```bash
# Inside your Python environment
from ultralytics import YOLO

# Load your model
model = YOLO('models/best.pt')

# Export to ONNX format with quantization
model.export(format='onnx', half=True)
```

Then use the quantized model:

```bash
python reefscape_yolo_api.py --model models/best_half.onnx --port 5000
```

### 2. NPU Acceleration

The Orange Pi 5 has an NPU (Neural Processing Unit) that can accelerate inference. You'll need to install the appropriate Rockchip NPU SDK and convert your model to a format compatible with the NPU.

For Rockchip RK3588S (Orange Pi 5):
1. Download and install the RKNN Toolkit
2. Convert your ONNX model to RKNN format
3. Modify the API service to use RKNN runtime instead of PyTorch

### 3. Resource Monitoring

Monitor resource usage to adjust batch size and image size for optimal performance:

```bash
# Install monitoring tools
sudo apt install -y htop iotop

# Monitor system while running inference
htop
```

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

### Prediction with Base64-encoded Image

```bash
# Encode image to base64
base64_img=$(base64 -w 0 test_image.jpg)

# Send request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$base64_img\", \"conf\":0.25}"
```

### Prediction with Image Path (Local Only)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_path\":\"/path/to/image.jpg\", \"conf\":0.25}"
```

## Troubleshooting

### Service Won't Start

Check the logs:
```bash
sudo journalctl -u yolov12-api
```

### Memory Issues

If the service crashes due to memory constraints:
1. Reduce model size using quantization
2. Use a smaller YOLOv12 variant (nano instead of small)
3. Reduce the image size parameter
4. Add swap memory:
   ```bash
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Slow Inference

1. Ensure you're using GPU/NPU acceleration
2. Check if any background processes are consuming resources
3. Reduce image resolution
4. Try model quantization
5. Consider using a more optimized runtime (ONNX Runtime, TensorRT, etc.)

## Security Considerations

The default configuration exposes the API to your local network. For production:

1. Add authentication:
   ```bash
   # Install Flask-HTTPAuth
   pip install Flask-HTTPAuth
   ```
   Then modify the code to require authentication.

2. Use HTTPS:
   ```bash
   # Install dependencies
   pip install pyopenssl
   ```
   Then modify the server code to use SSL context.

3. Restrict access to specific IP ranges in your network using a firewall. 