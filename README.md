# 📷 Detection Model

This repository contains a Python script for detecting and classifying shapes in images using the YOLOv8 model. The script processes images to identify shapes and classify them based on their contours.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How the Model Works](#how-the-model-works)
- [CLI Guide](#cli-guide)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/detection-model.git
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
    python Detection_model.py --image data.jpg --altitude 150 --confidence 0.5 --max_area 20
    ```

3. The script will load the image, process it, and classify the shapes found in the image.

## ⚙️ Configuration

- **Altitude**: The altitude in meters at which the image was taken. Default is `150`.
- **Confidence Threshold**: The confidence threshold for detection. Default is `0.5`.
- **Maximum Area to Filter**: The maximum area in meters squared to filter out shapes. Default is `20`.

These parameters can be adjusted directly in the `Detection_model.py` file or passed as arguments from the CLI.

## 🧠 How the Model Works

1. **Loading the Model**: The script starts by loading the YOLOv8 model using the `ultralytics` library.
    ```python
    model = YOLO('best.pt')
    ```

2. **Setting Parameters**: It sets various parameters such as altitude, confidence threshold, and maximum area to filter.
    ```python
    altitude = args.altitude
    confidence_threshold = args.confidence
    maximum_area_to_filter = args.max_area
    ```

3. **Loading the Image**: The script loads an image from the specified path.
    ```python
    image_path = args.image
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

## 🖥️ CLI Guide

To run the script from the command line, use the following command:

```sh
python Detection_model.py --image data.jpg --altitude 150 --confidence 0.5 --max_area 20