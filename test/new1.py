import cv2
import mediapipe as mp
import time

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Mediapipe setup for face detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Simulated Pan-Tilt control (for macOS)
def pan(angle):
    print(f"Pan angle set to {angle}")

def tilt(angle):
    print(f"Tilt angle set to {angle}")

# Light control function
def lights(r, g, b, w):
    for x in range(18):
        # Simulate light setup
        print(f"Setting light {x}: R={r} G={g} B={b} W={w}")

lights(0, 0, 0, 50)  # Initial light setup

# Constants for frame size
FRAME_W = 320
FRAME_H = 240

# Function to clamp values within a range
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# Track and update camera position dynamically
pTime = 0  # For calculating FPS
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        continue

    # Convert the frame to RGB (for Mediapipe)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(RGB_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape

            # Convert relative bounding box to absolute coordinates
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Draw rectangle and confidence score
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, f"ACCR: {int(detection.score[0] * 100)}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate the center of the face
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Calculate offset from frame center
            offset_x = face_center_x - FRAME_W // 2
            offset_y = face_center_y - FRAME_H // 2

            # Convert offsets to percentages
            turn_x = offset_x / (FRAME_W / 2)
            turn_y = offset_y / (FRAME_H / 2)

            # Scale offsets to degrees
            turn_x *= 10  # Adjust sensitivity
            turn_y *= 10

            # Update pan/tilt values
            cam_pan = 90 - turn_x
            cam_tilt = 90 + turn_y

            # Clamp values to servo range (0 to 180)
            cam_pan = clamp(cam_pan, 0, 180)
            cam_tilt = clamp(cam_tilt, 0, 180)

            # Simulate pan-tilt movement
            pan(cam_pan)
            tilt(cam_tilt)

            # Output face coordinates to the terminal
            print(f"Face moved: Coordinates (Center) - X: {face_center_x}, Y: {face_center_y}")

            # Display coordinates on the frame
            cv2.putText(frame, f"X: {face_center_x}, Y: {face_center_y}", 
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Break after the first face is detected (handle one face at a time)
            break

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Tracker", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
