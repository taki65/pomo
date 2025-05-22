import streamlit as st
import cv2
import numpy as np
from fer import FER
import mediapipe as mp
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sqlite3
import os

st.set_page_config(layout="wide")
st.title("🎯 감정/집중도 기반 맞춤형 뽀모도로 타이머")

# --- DB 초기화 ---
conn = sqlite3.connect("sessions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    angry REAL, disgust REAL, fear REAL, happy REAL, sad REAL,
    surprise REAL, neutral REAL, attention REAL, recommended_minutes REAL
)
''')
conn.commit()

# --- UI 설정 ---
st.markdown("## ⌚ 세션 시간 설정")
session_time = st.number_input("측정할 세션 시간 (분)", min_value=0.5, max_value=60.0, value=25.0, step=0.5)

# --- 기존 데이터 시각화 (입력 전만 표시) ---
if session_time == 25.0:  # 기본값 상태일 때만 표시
    st.subheader("📊 Previous Session Recommendation Trend")

    col_a, col_b = st.columns([4, 1])
    with col_a:
        df_hist = pd.read_sql_query("SELECT * FROM sessions", conn)
        if not df_hist.empty:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.plot(df_hist.index + 1, df_hist["recommended_minutes"], marker='o')
            ax_hist.set_xlabel("Session Number")
            ax_hist.set_ylabel("Recommended Time (min)")
            ax_hist.set_title("Recommendation Trend")
            st.pyplot(fig_hist)
        else:
            st.info("No saved session data. Start your first measurement!")
    with col_b:
        if st.button("🗑️ Reset Sessions"):
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            st.success("Session history has been cleared.")
            st.rerun()  #최신 스트림릿에 맞는 코드로 변경

        


# --- 모델 훈련 (CSV 기반) ---
@st.cache_resource
def train_model():
    df = pd.read_csv("synthetic_sessions.csv")
    X = df[df.columns[:-1]]
    y = df["recommended_minutes"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

if st.button("▶ 측정 시작"):
    emotion_detector = FER(mtcnn=False)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    data, timestamps, attn_scores = [], [], []

    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
        graph_placeholder = st.empty()
    with col2:
        emotion_placeholder = st.empty()
        result_box = st.empty()

    while cap.isOpened() and (time.time() - start_time) < session_time * 60:
        ret, frame = cap.read()
        if not ret:
            st.error("카메라 오류")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = face_mesh.process(img_rgb)

        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            top = emotions[0]
            (x, y, w_box, h_box) = top["box"]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            emotion_text = ", ".join([f"{k}: {v:.2f}" for k, v in top["emotions"].items()])
            emotion_dict = top["emotions"]
        else:
            emotion_text = "얼굴 감지 안됨"
            emotion_dict = {k: 0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}

        attention = 0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
            left_eye = np.mean([points[33], points[133]], axis=0)
            right_eye = np.mean([points[362], points[263]], axis=0)
            eye_center = (left_eye + right_eye) / 2
            screen_center = np.array([w / 2, h / 2])
            dist = np.linalg.norm(eye_center - screen_center)
            attention = max(0, 1 - dist / (w / 2))

        elapsed = time.time() - start_time
        timestamps.append(elapsed)
        attn_scores.append(attention)

        data.append({
            'timestamp': elapsed,
            **emotion_dict,
            'attention': attention
        })

        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_display, channels="RGB")

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 2.5)
        ax.plot(timestamps, attn_scores)
        ax.set_ylim(0, 1)
        ax.set_title("Real-time Attention")
        graph_placeholder.pyplot(fig, use_container_width=True)

        emotion_placeholder.markdown(f"**감정 상태**: {emotion_text}\n**집중도**: `{attention:.2f}`")

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(data)
    avg_attention = df["attention"].mean()
    df_grouped = df[['angry','disgust','fear','happy','sad','surprise','neutral','attention']].mean().to_frame().T

    # 모델 예측 및 저장
    recommended_time = round(float(model.predict(df_grouped)[0]), 2)
    result_box.success(f"✅ 추천 뽀모도로 시간: **{recommended_time}분**")

    cursor.execute("""
        INSERT INTO sessions (angry, disgust, fear, happy, sad, surprise, neutral, attention, recommended_minutes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(df_grouped.iloc[0]) + (recommended_time,))
    conn.commit()

    # CSV 파일에도 저장
    synthetic_path = "synthetic_sessions.csv"
    df_grouped["recommended_minutes"] = recommended_time
    if os.path.exists(synthetic_path):
        existing = pd.read_csv(synthetic_path)
        updated = pd.concat([existing, df_grouped], ignore_index=True)
        updated.to_csv(synthetic_path, index=False)
    else:
        df_grouped.to_csv(synthetic_path, index=False)