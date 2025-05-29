import cv2 as cv
import mediapipe as mp
import numpy as np
import os

dir= script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "crown.png")
crown = cv.imread(image_path, cv.IMREAD_UNCHANGED)
crown = cv.resize(crown, dsize=(0, 0), fx=0.1, fy=0.1)
h, w = crown.shape[:2]

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득 실패')
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = face_detection.process(rgb_frame)

    if res.detections:
        for det in res.detections:
            p = mp_face_detection.get_key_point(det, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            cx = int(p.x * frame.shape[1])
            cy = int(p.y * frame.shape[0]) - h  # 원래 위치에서 crown 높이만큼 위로 이동

            x1 = max(cx - w // 2, 0)
            y1 = max(cy - h // 2, 0)
            x2 = min(cx + w // 2, frame.shape[1])
            y2 = min(cy + h // 2, frame.shape[0])

            roi = frame[y1:y2, x1:x2]

            # 리사이즈한 crown 이미지 및 alpha 채널
            crown_resized = cv.resize(crown[:, :, :3], (x2 - x1, y2 - y1))
            alpha = cv.resize(crown[:, :, 3], (x2 - x1, y2 - y1))
            alpha = alpha.astype(float) / 255.0
            alpha = alpha[:, :, np.newaxis]  # (h, w, 1)

            # roi와 crown 합성 (float 연산 후 uint8 변환)
            roi = roi.astype(float)
            crown_resized = crown_resized.astype(float)

            blended = roi * (1 - alpha) + crown_resized * alpha
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    cv.imshow('MediaPipe FACE AR', cv.flip(frame, 1))
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()