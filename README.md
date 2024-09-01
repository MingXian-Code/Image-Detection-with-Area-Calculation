# 📷 Detection Model

This repository contains Python scripts for detecting and classifying shapes in images using the YOLOv8 model, as well as training the YOLOv8 model. The scripts process images to identify shapes and classify them based on their contours.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How the Model Works](#how-the-model-works)
- [Training the Model](#training-the-model)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MingXian-Code/detection-model.git
    cd detection-model
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 🚀 Usage

1. Place the image you want to process in the root directory of the project and name it `data.jpg`.

2. Run the script:
    ```sh
    python Detection_model.py
    ```

3. The script will load the image, process it, and classify the shapes found in the image.

## ⚙️ Configuration

- **Distance**: The distance in meters at which the image was taken. Default is `150`.
- **Confidence Threshold**: The confidence threshold for detection. Default is `0.5`.
- **Maximum Area to Filter**: The maximum area in meters squared to filter out shapes. Default is `20`.

These parameters can be adjusted directly in the `Detection_model.py` file.

## 🧠 How the Model Works

1. **Loading the Model**: The script starts by loading the YOLOv8 model using the `ultralytics` library.
    ```python
    model = YOLO('best.pt')
    ```

2. **Setting Parameters**: It sets various parameters such as distance, confidence threshold, and maximum area to filter.
    ```python
    distance = 150  # distance in meters
    confidence_threshold = 0.5  # Confidence threshold for detection
    Maximum_area_to_filter = 20  # Maximum area in meters squared to filter out shapes
    ```

3. **Loading the Image**: The script loads an image from the specified path.
    ```python
    image_path = 'data.jpg'
    image = cv2.imread(image_path)
    ```

4. **Defining Camera FOV and Image Dimensions**: It defines the camera's field of view (FOV) and image dimensions.
    ```python
    fov_horizontal = 62.2  # Horizontal field of view in degrees
    fov_vertical = 48.8  # Vertical field of view in degrees (example value)
    image_width = 3280  # Image width in pixels
    image_height = 2464  # Image height in pixels
    ```

5. **Classifying Shapes**: The script includes a function to classify shapes based on their contours.
    ```python
    def classify_shape(contour):
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.0385 * peri, True)  # Tighten the approximation for quadrilaterals
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "quadrilateral"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        else:
            shape = "circle"
        return shape
    ```

## 🚀 Training the Model

This repository also contains a script to train a YOLOv8 model using the `ultralytics` library. The script allows you to specify various parameters such as the model file, data configuration, number of epochs, device, and batch size.

### Requirements

- Python 3.x
- `ultralytics` library
- `argparse` library (standard with Python)

### Installation

1. Clone the repository:
    ```sh
    git clone  https://github.com/MingXian-Code/Model_training_local_machine.git
    cd training
    ```

2. Install the required libraries:
    ```sh
    pip install ultralytics
    ```

### Usage

To train the YOLOv8 model, run the `Model_training_local_machine.py` script with the appropriate arguments.

#### Arguments

- `--model`: Path to the YOLO model file (default: `yolov8l.pt`).
- `--data`: Path to the data configuration file (required).
- `--epochs`: Number of training epochs (default: `100`).
- `--device`: Device to use for training (e.g., `"0"` for GPU, `"cpu"` for CPU) (default: `"0"`) (Make sure your GPU is CUDA capable).
- `--batch`: Batch size for training (default: `4`).

#### Example

```sh
python Model_training_local_machine.py --data path/to/data.yaml --epochs 50 --device 0 --batch 8
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
