import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Use a YOLOv8 model, e.g., 'yolov8n.pt' for Nano version.

def calculate_dominance_score(face_area, eye_stability, mouth_movement):
    """
    Calculate a dominance score based on normalized face area, eye stability, and mouth movement.
    """
    normalized_face_area = face_area / 40000  # Adjust based on resolution
    dominance_score = normalized_face_area * 0.5 + eye_stability * 0.3 + mouth_movement * 0.2
    return dominance_score

def detect_and_track_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read the frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Run YOLO detection on the frame
        results = model(frame)

        dominant_face = None
        dominant_score = -1
        dominant_coords = (0, 0)

        for result in results:  # Iterate through the detections
            for detection in result.boxes:  # Each result contains bounding boxes
                # Extract bounding box details
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
                confidence = float(detection.conf[0])  # Confidence score
                cls = int(detection.cls[0])  # Class label

                # Check if the detection corresponds to a person or face class
                if confidence > 0.5 and cls == 0:  # Assuming class 0 is "person"
                    # Calculate face area and center
                    w, h = x2 - x1, y2 - y1
                    face_area = w * h
                    face_center = (x1 + w // 2, y1 + h // 2)

                    # Eye and Mouth Tracking (placeholder values; you can replace with real tracking logic)
                    eye_stability, mouth_movement = 0, 0

                    # Calculate dominance score
                    dominance_score = calculate_dominance_score(face_area, eye_stability, mouth_movement)

                    # Update dominant face
                    if dominance_score > dominant_score:
                        dominant_score = dominance_score
                        dominant_face = face_center
                        dominant_coords = (x1, y1, w, h)

                    # Draw bounding box and score
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Coords: {face_center}, Score: {dominance_score:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Highlight dominant face
        if dominant_face:
            x, y, w, h = dominant_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, f"Dominant Face - Coords: {dominant_face}, Score: {dominant_score:.2f}",
                        (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Print coordinates and score to the terminal
            print(f"Dominant Face Coords: {dominant_face}, Dominance Score: {dominant_score:.2f}")

        # Display the frame
        cv2.imshow("Dominant Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_track_faces()
