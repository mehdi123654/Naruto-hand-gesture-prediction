import cv2
import time
from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO("../models/naruto_jutsu_detector11/weights/best.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Record start time
        start_time = time.time()

        # Run detection
        results = model.predict(frame, imgsz=640, conf=0.7, device=0, verbose=False)

        # Visualize detections
        annotated_frame = results[0].plot()

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Draw FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Naruto Jutsu Detector", annotated_frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
