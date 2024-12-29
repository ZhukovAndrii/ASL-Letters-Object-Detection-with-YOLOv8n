import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('best2.pt')  # Ensure 'best.pt' is in the same directory as this script

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform inference on the current frame
    results = model.predict(frame, conf=0.4, show=False)  # Adjust conf as needed

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels

    # Display the frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
