import cv2
import efficientdet

# Load the EfficientDet model
model = efficientdet.load_model('efficientdet_d0')

# Capture video from the laptop camera
cap = cv2.VideoCapture(0)

while True:
    # Capture the next frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to the model's input size
    frame_resized = cv2.resize(frame_rgb, (512, 512))

    # Convert the resized frame to a NumPy array
    frame_array = np.expand_dims(frame_resized, axis=0)

    # Make predictions on the frame
    predictions = model.predict(frame_array)

    # Extract the bounding boxes and class labels from the predictions
    boxes = predictions['detection_boxes']
    classes = predictions['detection_classes']
    scores = predictions['detection_scores']

    # Filter out detections with low confidence scores
    boxes, classes, scores = filter_predictions(boxes, classes, scores, score_threshold=0.5)

    # Draw bounding boxes and class labels on the frame
    for box, class_, score_ in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'{class_}: {score_:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
