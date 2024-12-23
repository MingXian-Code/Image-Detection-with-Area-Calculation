import torch
from ultralytics import YOLO

# Load the PyTorch model
model = YOLO('yolov8s.pt')  # Load the YOLOv8 model from the ultralytics library

# Dummy input for the model
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX format
torch.onnx.export(model.model, dummy_input, 'yolov8s.onnx', opset_version=11)