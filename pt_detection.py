import cv2
import numpy as np
from ultralytics import YOLO

# Loadmodel
model = YOLO('V:/coding/dev/Python/Gantry/FYP/best.pt')

# Function to perform detection on an image
def detect_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Perform inference
    results = model(image)

    # Extract results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    class_names = results[0].names  # Class names

    # Draw the results on the image
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > 0.2:  # Threshold for displaying boxes
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            label = f"{class_name}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output image saved at: {output_path}")

input_image_path = 'V:/coding/dev/Python/Gantry/FYP/raw/resized/20241006_173537.jpg'
output_image_path = 'V:/coding/dev/Python/Gantry/FYP/output/20241006_173537_detected.jpg'
detect_image(input_image_path, output_image_path)