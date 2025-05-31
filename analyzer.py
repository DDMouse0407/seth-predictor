
import pandas as pd
import numpy as np
import datetime
import random
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假資料模擬函式
def simulate_data(n=20):
    today = datetime.date.today()
    data = []
    for i in range(n):
        date = today - datetime.timedelta(days=i)
        plays = random.randint(30, 100)
        free_game_triggered = random.randint(0, 1)
        small_hit = random.randint(0, 1)
        jackpot = 1 if free_game_triggered and random.random() > 0.7 else 0
        burst_index = (plays * 0.05 + small_hit * 10 + jackpot * 50) / 100
        data.append({
            "日期": date.strftime('%Y-%m-%d'),
            "局數": plays,
            "免費遊戲": free_game_triggered,
            "小分": small_hit,
            "爆金": jackpot,
            "爆發指數": round(burst_index, 2)
        })

    df = pd.DataFrame(data[::-1])  # 時間升序排列
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/history.csv", index=False)
    return df

# 最新資料分析邏輯（可擴充）
def analyze_latest_data():
    try:
        df = pd.read_csv("data/history.csv")
        latest = df.iloc[-1]
        msg = f"最近日期：{latest['日期']}，局數：{latest['局數']}，爆發指數：{latest['爆發指數']}"
        return msg
    except Exception as e:
        return f"資料讀取失敗：{str(e)}"

# AI 模型訓練函式
def train_xgb_model():
    try:
        df = pd.read_csv("data/history.csv")
        X = df[["局數", "免費遊戲", "小分", "爆發指數"]]
        y = df["爆金"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        os.makedirs("model", exist_ok=True)
        model.save_model("model/xgb_model.json")
        return f"✅ 模型訓練完成，準確率：{acc*100:.2f}%"
    except Exception as e:
        return f"❌ 模型訓練失敗：{str(e)}"

# 模型預測函式
def predict_jackpot(input_data):
    try:
        model = xgb.XGBClassifier()
        model.load_model("model/xgb_model.json")
        prediction = model.predict(pd.DataFrame([input_data]))
        prob = model.predict_proba(pd.DataFrame([input_data]))[0][1]
        return int(prediction[0]), prob
    except Exception as e:
        return -1, f"預測失敗：{str(e)}"
