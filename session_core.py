import cv2
import time
import numpy as np
import pandas as pd
from fer import FER
import mediapipe as mp
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt

# --- 감정 인식 ---
emotion_detector = FER(mtcnn=False)

# --- 얼굴 메쉬 ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# --- 집중도 추정 ---
def estimate_attention(frame, landmarks, w, h):
    points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
    left_eye_center = np.mean([points[33], points[133]], axis=0)
    right_eye_center = np.mean([points[362], points[263]], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2
    screen_center = np.array([w / 2, h / 2])
    dist = np.linalg.norm(eye_center - screen_center)
    max_dist = w / 2
    attention_score = max(0, 1 - dist / max_dist)
    return attention_score

# --- 실시간 감정/집중도 측정 + 시각화 ---
def analyze_session(duration_minutes=1, output_file='session_data.csv'):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    start_time = time.time()
    data = []
    timestamps = []
    attention_scores = []

    print(f"세션 시작: {duration_minutes}분 동안 측정 중...")

    while (time.time() - start_time) < duration_minutes * 60:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 오류")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            top_emotions = emotions[0]["emotions"]
            (x, y, w_box, h_box) = emotions[0]["box"]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        else:
            top_emotions = {k: 0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}

        attention = 0
        h, w, _ = frame.shape
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # 랜드마크 점 찍는 코드 제거:
            # mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            attention = estimate_attention(frame, landmarks, w, h)

        elapsed = time.time() - start_time
        timestamps.append(elapsed)
        attention_scores.append(attention)

        data.append({
            'timestamp': elapsed,
            **top_emotions,
            'attention': attention
        })

        info = f"Attn: {attention:.2f} | Emotions: {max(top_emotions, key=top_emotions.get)}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("감정 + 집중도 측정", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    actual_duration = (time.time() - start_time) / 60
    print(f"측정 종료, 저장 완료: {output_file}")
    print(f"실제 시간: {actual_duration:.2f}분")
    return actual_duration

# --- 쉬는 시간 타이머 ---
def break_timer(minutes=5):
    print(f"\n쉬는 시간 {minutes}분 시작...")
    time.sleep(minutes * 60)
    print("쉬는 시간 종료!\n")

# --- 회귀모델로 다음 시간 추천 ---
def train_regression_model(data_path='session_data.csv', all_sessions_path='sessions_all.csv'):
    df = pd.read_csv(data_path)
    feature_cols = ['angry','disgust','fear','happy','sad','surprise','neutral','attention']
    df_grouped = df[feature_cols].mean().to_frame().T
    df_grouped['recommended_time'] = 25

    if os.path.exists(all_sessions_path):
        prev = pd.read_csv(all_sessions_path)
        full = pd.concat([prev, df_grouped], ignore_index=True)
    else:
        full = df_grouped

    full.to_csv(all_sessions_path, index=False)

    if len(full) > 3:
        X = full[feature_cols]
        y = full['recommended_time']
        model = LinearRegression()
        model.fit(X, y)
        new_time = model.predict(df_grouped[feature_cols])[0]
        print(f"추천 시간: {round(new_time,2)}분")
        return max(5, round(new_time, 2))
    else:
        print("데이터 부족 → 기본값 유지")
        return 25

# --- 사용자에게 다음 시간 입력받기 ---
def ask_next_duration(default_duration):
    try:
        user_input = input(f"다음 뽀모도로 시간? (기본값 {default_duration}): ").strip()
        if user_input == "":
            return default_duration
        val = float(user_input)
        if val < 5:
            print("최소 5분 이상 입력하세요.")
            return default_duration
        return val
    except:
        print("잘못된 입력. 기본값 사용.")
        return default_duration

# --- 전체 실행 함수 (UI에서 호출할 메인 엔트리) ---
def run_pomodoro_session(initial_duration=1, break_duration=0.17):
    actual_duration = analyze_session(duration_minutes=initial_duration, output_file='session_data.csv')
    break_timer(minutes=break_duration)
    
    recommended_duration = train_regression_model(data_path='session_data.csv')
    print(f"\n📈 추천된 다음 뽀모도로 시간: {recommended_duration}분")

    # 사용자 입력 반영
    next_duration = ask_next_duration(recommended_duration)
    print(f"⏱ 다음 세션 시간: {next_duration}분으로 설정됨.\n")
    
    return next_duration

def ask_next_duration(default_duration):
    try:
        user_input = input(f"\n다음 뽀모도로 시간을 입력하세요 (기본값 {default_duration}분, 최소 0.5분): ").strip()
        if user_input == "":
            return default_duration
        val = float(user_input)
        if val < 0.5:
            print("❗ 최소 0.5분 이상 입력해야 합니다.")
            return default_duration
        return val
    except:
        print("❗ 잘못된 입력입니다. 기본값으로 진행합니다.")
        return default_duration