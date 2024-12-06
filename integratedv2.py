import cv2
import mediapipe as mp
import numpy as np

# Mediapipe initializations
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read the frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(rgb_frame)
            dominant_face = None
            dominant_score = -1
            dominant_coords = (0, 0)

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
                    face_center = (x + w // 2, y + h // 2)

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

                            left_eye_stability = np.std([pt[1] for pt in left_eye_coords])
                            right_eye_stability = np.std([pt[1] for pt in right_eye_coords])
                            eye_stability = max(0, 10 - (left_eye_stability + right_eye_stability) / 2)

                            # Mouth Tracking
                            mouth_indices = [78, 308, 13, 14, 87, 317]
                            mouth_coords = [(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                                            for i in mouth_indices]

                            mouth_diff = np.mean([abs(pt[1] - mouth_coords[0][1]) for pt in mouth_coords])
                            mouth_movement = min(15, mouth_diff)

                    # Calculate dominance score
                    dominance_score = calculate_dominance_score(face_area, eye_stability, mouth_movement)

                    # Update dominant face
                    if dominance_score > dominant_score:
                        dominant_score = dominance_score
                        dominant_face = face_center
                        dominant_coords = (x, y, w, h)

                    # Display all face coordinates and scores
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Coords: {face_center}, Score: {dominance_score:.2f}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
