import cv2
import tensorflow
import numpy as np
from ultralytics import YOLO

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="V:/coding/dev/Python/Gantry/FYP/best_float32.tflite")
interpreter.allocate_tensors()

#input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image_path = "V:/coding/dev/Python/Gantry/FYP/resized/20241006_131544.jpg"
image = cv2.imread(image_path)
input_shape = input_details[0]['shape']
input_data = cv2.resize(image, (input_shape[1], input_shape[2]))
input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0

# inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Extract and process predictions
detections = interpreter.get_tensor(output_details[0]['index'])

# Process the predictions
for detection in detections[0]:  # Iterate through detected objects
    confidence = detection[4]
    if confidence > 0.1:  # Confidence threshold
        xmin, ymin, xmax, ymax = detection[0:4]
        (left, right, top, bottom) = (int(xmin * image.shape[1]), int(xmax * image.shape[1]),
                                      int(ymin * image.shape[0]), int(ymax * image.shape[0]))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        class_id = int(detection[5])
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display and save the image with detections
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("V:/coding/dev/Python/Gantry/FYP/output/16output_image.jpg", image)