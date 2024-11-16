import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('V:/coding/dev/Python/Gantry/FYP/best.pt')

# Function to perform detection on a camera feed
def detect_camera():
    '''function to perform detection on a camera feed'''
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read() # Read the frame
        if not ret:
            print("Error: Could not read frame.")
            break
        results = model(frame) # Perform inference on the frame

        # Extract results
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
        class_names = results[0].names  # Class names

        # Draw the results on the frame
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.2:  # Threshold for displaying boxes
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                label = f"{class_name}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4) # Draw the bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Draw the label

        # Display the resulting frame
        cv2.imshow('YOLOv8 Detection', frame)

        # Press 'q' to break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
    cap.release()
    cv2.destroyAllWindows()

# Run the camera detection
detect_camera()
