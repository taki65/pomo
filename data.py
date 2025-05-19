import pandas as pd
import numpy as np
import os



# --- 가상 데이터 생성 ---
path="synthetic_sessions.csv"
if not os.path.exists(path):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "angry": np.random.beta(2, 5, n),
        "disgust": np.random.beta(1, 8, n),
        "fear": np.random.beta(2, 6, n),
        "happy": np.random.beta(5, 2, n),
        "sad": np.random.beta(2, 5, n),
        "surprise": np.random.beta(2, 4, n),
        "neutral": np.random.beta(4, 3, n),
        "attention": np.random.uniform(0.4, 1.0, n)
    })
    weights = np.array([0.5, 0.2, 0.3, -0.7, 0.6, 0.4, -0.2, 1.2])
    noise = np.random.normal(0, 2, n)
    df["recommended_minutes"] = 30 + df[df.columns].values @ weights + noise
    df["recommended_minutes"] = np.clip(df["recommended_minutes"], 20, 60)
    df.to_csv(path, index=False)
