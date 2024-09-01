import argparse
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model.')
    parser.add_argument('--model', type=str, default="yolov8l.pt", help='Path to the YOLO model file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data configuration file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--device', type=str, default="0", help='Device to use for training (e.g., "0" for GPU, "cpu" for CPU).')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the model
    model = YOLO(args.model)

    # Use the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        device=args.device,
        batch=args.batch
    )

if __name__ == "__main__":
    main()