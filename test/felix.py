import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Mediapipe initializations
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Rolling average storage for total focus value
total_focus_history = deque(maxlen=3)

def normalize(value, min_val, max_val):
    """
    Normalize a value to a 0–100 range.
    """
    return int(100 * (value - min_val) / (max_val - min_val)) if max_val > min_val else 0

def calculate_focus(face_area, eye_stability, mouth_movement):
    """
    Calculate normalized focus values and total focus.
    """
    max_face_area = 40000  # Adjust for camera resolution
    face_focus = normalize(face_area, 0, max_face_area)
    eye_focus = normalize(eye_stability, 0, 10)  # Eye stability in a 0–10 range
    mouth_focus = normalize(mouth_movement, 0, 15)  # Mouth movement intensity in a 0–15 range

    # Smooth total focus with rolling average
    total_focus = (face_focus + eye_focus + mouth_focus) // 3
    total_focus_history.append(total_focus)
    smoothed_total_focus = sum(total_focus_history) // len(total_focus_history)

    return face_focus, eye_focus, mouth_focus, smoothed_total_focus

def detect_faces_and_focus():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read the frame.")
                break

            # Flip for a mirrored view
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(rgb_frame)

            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, w, h = (
                        int(bboxC.xmin * width),
                        int(bboxC.ymin * height),
                        int(bboxC.width * width),
                        int(bboxC.height * height),
                    )
                    face_area = w * h

                    # Eye and Mouth Tracking
                    face_mesh_results = face_mesh.process(rgb_frame)
                    eye_stability, mouth_movement = 0, 0

                    if face_mesh_results.multi_face_landmarks:
                        for landmarks in face_mesh_results.multi_face_landmarks:
                            # Eye Tracking
                            left_eye_indices = [33, 160, 158, 133, 153, 144]
                            right_eye_indices = [362, 385, 387, 263, 373, 380]

                            left_eye_coords = [(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                                               for i in left_eye_indices]
                            right_eye_coords = [(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                                                for i in right_eye_indices]

                            # Calculate eye stability
                            left_eye_stability = np.std([pt[1] for pt in left_eye_coords])
                            right_eye_stability = np.std([pt[1] for pt in right_eye_coords])
                            eye_stability = max(0, 10 - (left_eye_stability + right_eye_stability) / 2)  # Stability score

                            # Draw eye landmarks
                            for pt in left_eye_coords + right_eye_coords:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

                            # Mouth Tracking
                            mouth_indices = [78, 308, 13, 14, 87, 317]
                            mouth_coords = [(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                                            for i in mouth_indices]

                            mouth_diff = np.mean([abs(pt[1] - mouth_coords[0][1]) for pt in mouth_coords])
                            mouth_movement = min(15, mouth_diff)  # Scale movement to max 15

                            # Draw mouth landmarks
                            for pt in mouth_coords:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

                    # Calculate focus values
                    face_focus, eye_focus, mouth_focus, total_focus = calculate_focus(
                        face_area, eye_stability, mouth_movement
                    )

                    # Draw bounding box and display focus values
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    focus_text = f"Face: {face_focus}, Eye: {eye_focus}, Mouth: {mouth_focus}, Total: {total_focus}"
                    cv2.putText(frame, focus_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Focus Detection", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting focus detection system...")
    detect_faces_and_focus()