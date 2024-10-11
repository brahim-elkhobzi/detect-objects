import cv2
from ultralytics import YOLO

# Load the YOLOv8 model 
model = YOLO('yolov8n.pt')  


cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously get frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv8 object detection on the frame
    results = model(frame)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the frame with detected objects
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press 'q' to exit the webcam stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
