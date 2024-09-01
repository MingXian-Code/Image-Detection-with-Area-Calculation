import numpy as np
import cv2
import os
import argparse
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect and classify shapes in an image using YOLOv8.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--distance', type=float, default=150, help='Distance in meters at which the image was taken.')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detection.')
    parser.add_argument('--max_area', type=float, default=20, help='Maximum area in meters squared to filter out shapes.')
    return parser.parse_args()

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

def calculate_area(contour, shape):
    if shape in ["triangle", "quadrilateral", "pentagon", "hexagon"]:
        return cv2.contourArea(contour)
    elif shape == "circle":
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        return np.pi * (radius ** 2)
    else:
        return 0

def convert_area_to_meters(area_pixels, distance, fov_horizontal, fov_vertical, image_width, image_height):
    # Calculate the ground sampling distance (GSD) for both dimensions
    gsd_x = (2 * distance * np.tan(np.deg2rad(fov_horizontal / 2))) / image_width
    gsd_y = (2 * distance * np.tan(np.deg2rad(fov_vertical / 2))) / image_height
    area_meters = area_pixels * (gsd_x * gsd_y)
    return area_meters

def main():
    args = parse_arguments()

    # Load the YOLOv8 model
    model = YOLO('best.pt')

    Distance = args.distance
    confidence_threshold = args.confidence
    maximum_area_to_filter = args.max_area

    # Load an image
    image_path = args.image
    image = cv2.imread(image_path)

    # Define the camera's field of view (FOV) and image dimensions
    fov_horizontal = 62.2  # Horizontal field of view in degrees
    fov_vertical = 48.8  # Vertical field of view in degrees (example value)
    image_width = 3280  # Image width in pixels
    image_height = 2464  # Image height in pixels

    # Perform detection with a confidence threshold
    results = model(image, conf=confidence_threshold)

    # Extract bounding boxes and class labels
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = model.names  # Access class names directly from the model

    # Calculate areas in pixels squared and meters squared
    areas_pixels = []
    areas_meters = []
    color_areas = {}
    color_counts = {}

    # Create results directory if it doesn't exist
    results_dir = 'result_wt'
    os.makedirs(results_dir, exist_ok=True)

    # Print the areas and class names
    for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            continue  # Skip invalid bounding boxes

        roi = image[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            continue  # Skip empty ROIs

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            shape = classify_shape(contour)
            area_pixels = calculate_area(contour, shape)
            area_meters = convert_area_to_meters(area_pixels, Distance, fov_horizontal, fov_vertical, image_width, image_height)
            
            if area_meters > maximum_area_to_filter:
                continue  # Skip shapes with area greater than x meters squared
            
            areas_pixels.append(area_pixels)
            areas_meters.append(area_meters)
            
            label = class_names[int(class_id)]
            print(f"Shape {i+1}: {label}, Area: {area_pixels} pixels², {area_meters:.2f} meters²")
            
            # Draw bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (int(x1) + x, int(y1) + y), (int(x1) + x + w, int(y1) + y + h), (0, 255, 0), 2)
            
            # Draw the shape number and label on the bounding box
            text = f"{i+1}: {label}"
            cv2.putText(image, text, (int(x1) + x, int(y1) + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Draw the contour on the image
            cv2.drawContours(image, [contour + np.array([int(x1), int(y1)])], -1, (0, 0, 255), 2)
            
            # Update the total area and count for each color
            if label not in color_areas:
                color_areas[label] = 0
                color_counts[label] = 0
            color_areas[label] += area_meters
            color_counts[label] += 1
            
    # Calculate the total area in pixels squared and meters squared
    total_area_pixels = sum(areas_pixels)
    total_area_meters = sum(areas_meters)

    # Print the total area and number of detected objects
    print(f"Total Area: {total_area_pixels} pixels², {total_area_meters:.2f} meters²")
    print(f"Total number of objects detected: {len(areas_pixels)}")

    # Save the image with bounding boxes, labels, and contours
    output_image_path = os.path.join(results_dir, f'result_{Distance}m.jpg')
    cv2.imwrite(output_image_path, image)

    # Save the results to a text file
    results_file_path = os.path.join(results_dir, 'results.txt')
    with open(results_file_path, 'w') as f:
        f.write(f"At distance {Distance} meters:\n")
        f.write(f"Total area of detected shapes: {total_area_pixels} pixels²\n")
        f.write(f"Total area of detected shapes: {total_area_meters:.2f} meters²\n")
        f.write(f"Total number of objects detected: {len(areas_pixels)}\n")
        f.write("\n")
        f.write("Individual shape details (excluding shapes with area > 50 meters²):\n")
        for i, (area_pixels, area_meters) in enumerate(zip(areas_pixels, areas_meters)):
            label = class_names[int(class_ids[i])]
            f.write(f"Shape {i+1}: {label}, Area: {area_pixels} pixels², {area_meters:.2f} meters²\n")
        f.write("\n")
        f.write("Total area and count by color:\n")
        for color, area in color_areas.items():
            count = color_counts[color]
            f.write(f"{color}: {area:.2f} meters², Quantity: {count}\n")

    # Display the image with bounding boxes, labels, and contours
    cv2.imshow('Detected Shapes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()