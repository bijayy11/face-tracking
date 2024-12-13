import cv2
import mediapipe as mp
import time


left_step_limit = 0
right_step_limit = 6000
current_step = 3000

def calculate_steps_for_motor(bbox_center_x, frame_center_x, iw, max_steps=6000):
    # Calculate the offset between the bounding box center and frame center
    offset = bbox_center_x - frame_center_x
   
    # Calculate number of steps the motor should move
    steps_to_move = int((offset / iw) * max_steps)

    return steps_to_move


def move_motor(steps):
    global current_step
   
    # Update current step position based on the number of steps to move
    current_step += steps
    # Ensure the current_step is within the defined limits
    if current_step < left_step_limit:
        current_step = left_step_limit
    elif current_step > right_step_limit:
        current_step = right_step_limit
   
    # Print the number of steps to be moved and the current_step value
    print(f"Moving motor by {steps} steps. Current step: {current_step}.")

# Function to calculate mouth openness
def calculate_mouth_open(face_landmarks, ih, iw):
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    mouth_open_score = abs(lower_lip.y - upper_lip.y) * ih
    return mouth_open_score

# Function to calculate distance (based on bounding box size)
def calculate_distance(face, ih, iw):
    bboxC = face.location_data.relative_bounding_box
    area = bboxC.width * bboxC.height
    return area

# Function to calculate hand gesture score
def calculate_hand_gesture(hands, ih, iw):
    if hands.multi_hand_landmarks:
        return 1.0  # Example: Presence of hands gives max score (implement specific gestures if needed)
    return 0.0

# Function to calculate eye focus score
def calculate_eye_focus(face_landmarks):
    left_eye = face_landmarks[145]
    right_eye = face_landmarks[374]
    eye_focus_score = abs(left_eye.y - right_eye.y)  # Placeholder for an actual eye focus algorithm
    return eye_focus_score

# Function to calculate dominant score
def calculate_dominant_score(distance, mouth_open, hand_gesture, eye_focus, distance_factor, mouth_open_factor, hand_gesture_factor, eye_focus_factor):
    score = (distance_factor * distance +
             mouth_open_factor * mouth_open +
             hand_gesture_factor * hand_gesture +
             eye_focus_factor * eye_focus)
    return score

def main():
    # Initialize MediaPipe solutions
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Factors for dominance calculation
    distance_factor = 0.2
    mouth_open_factor = 0.6
    hand_gesture_factor = 0.1
    eye_focus_factor = 0.1

    # Initialize video capture (0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get the resolution of the camera
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    # Use MediaPipe Face Detection, Face Mesh, and Hands
    with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as face_mesh, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:

        dominant_face = None
        last_update_time = 0

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Mirror the frame (flip horizontally)
            frame = cv2.flip(frame, 1)
            ih, iw, _ = frame.shape

            # Calculate the center of the frame
            frame_center_x = iw // 2
            frame_center_y = ih // 2

            # Display the center coordinates on the frame
            cv2.putText(frame, f"Center: ({frame_center_x}, {frame_center_y})", (frame_center_x - 80, frame_center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw the center of the frame (a small circle at the center)
            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)  # Red dot

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform face detection
            results_face = face_detection.process(frame_rgb)

            # Perform face mesh
            results_mesh = face_mesh.process(frame_rgb)

            # Perform hand detection
            results_hands = hands.process(frame_rgb)

            if results_face.detections and results_mesh.multi_face_landmarks:
                faces = results_face.detections
                face_landmarks = results_mesh.multi_face_landmarks[0].landmark

                # Calculate dominant face every 1 second
                current_time = time.time()
                if current_time - last_update_time > 1:
                    max_score = -1
                    for face in faces:
                        distance = calculate_distance(face, ih, iw)
                        mouth_open = calculate_mouth_open(face_landmarks, ih, iw)
                        hand_gesture = calculate_hand_gesture(results_hands, ih, iw)
                        eye_focus = calculate_eye_focus(face_landmarks)

                        score = calculate_dominant_score(distance, mouth_open, hand_gesture, eye_focus,
                                                         distance_factor, mouth_open_factor, hand_gesture_factor, eye_focus_factor)
                        if score > max_score:
                            max_score = score
                            dominant_face = face
                    last_update_time = current_time

                # Draw only the dominant face
                if dominant_face:
                    bboxC = dominant_face.location_data.relative_bounding_box
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Calculate the center of the bounding box
                    bbox_center_x = x + w // 2
                    bbox_center_y = y + h // 2

                    # Draw the bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display the center coordinates on the bounding box
                    cv2.putText(frame, f"Center: ({bbox_center_x}, {bbox_center_y})", (bbox_center_x - 40, bbox_center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Calculate the number of steps the motor should move
                    steps = calculate_steps_for_motor(bbox_center_x, frame_center_x, iw)
                    move_motor(steps)

                    # Move the motor based on the calculated steps
                    move_motor(steps)

                    print(f"Dominant face at: x={x}, y={y}, width={w}, height={h}, center=({bbox_center_x}, {bbox_center_y})")

            # Display the resulting frame
            cv2.imshow('Face Tracking', frame)

            # Break the loop on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()