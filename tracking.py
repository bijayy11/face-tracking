import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while cap.isOpened():
    ret, frame = cap.read()
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(RGBframe)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            mpDraw.draw_detection(frame, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, f"ACCR: {str(int(detection.score[0] * 100))}%", (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2,
                        cv2.LINE_AA)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("opt", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()