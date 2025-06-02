# 🎯 감정/집중도 기반 맞춤형 뽀모도로 타이머

실시간으로 **얼굴 감정**과 **집중도**를 분석하고, 사용자의 상태에 맞춰 **최적의 뽀모도로 시간**을 머신러닝으로 추천해주는 스마트 타이머입니다.

![demo](https://img.shields.io/badge/기반-Mediapipe%20%7C%20FER%20%7C%20Streamlit-blue)

---

## 📌 주요 기능

- 📸 웹캠 기반 실시간 **감정 인식**
- 👀 **시선 추적**을 통한 집중도 계산
- ⏱️ 세션 동안의 평균 감정/집중도 기록
- 📈 머신러닝을 활용한 **추천 시간 출력**
- 🧠 사용자 맞춤형 뽀모도로 루프
- 📊 이전 세션 그래프 확인 및 초기화

---

## 🧰 설치 및 실행 방법

main.py 실행은 경로이동후 터미널에서 uvicorn main:app 으로 실행후 웹경로는 127.0.0.1:8000 접근
### 1️⃣ 환경 구성

```bash
# 가상 환경 생성
conda create -n pomo python=3.9

# 가상 환경 활성화
conda activate pomo

# 필요 패키지 설치
pip install -r requirements.txt


🧠 사용 기술
	•	[FER] 얼굴 감정 분석
	•	[MediaPipe] 얼굴 랜드마크 및 시선 추적
	•	[OpenCV] 이미지 처리 및 웹캠 제어
	•	[Streamlit] 인터랙티브 웹 앱
	•	[Scikit-learn] 머신러닝 모델 학습
	•	[SQLite3] 간편한 세션 데이터 저장

 