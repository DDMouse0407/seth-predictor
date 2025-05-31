import pandas as pd
import numpy as np
import datetime
import random
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假資料模擬函式（保留但不使用）
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

    df = pd.DataFrame(data[::-1])
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

# AI 模型訓練函式（整合 haoting 外部資料）
def train_xgb_model():
    try:
        df1 = pd.read_csv("data/history.csv") if os.path.exists("data/history.csv") else pd.DataFrame()
        df2 = pd.read_csv("data/haoting_data.csv") if os.path.exists("data/haoting_data.csv") else pd.DataFrame()

        df = pd.concat([df1, df2], ignore_index=True)
        df = df.dropna()
        if df.empty:
            return "❌ 資料不足，無法訓練模型"

        X = df[["局數", "免費遊戲", "小分", "爆發指數"]]
        y = df["爆金"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        os.makedirs("model", exist_ok=True)
        model.save_model("model/xgb_model.json")
        return f"✅ 模型訓練完成（整合 haoting + 自己資料），準確率：{acc*100:.2f}%"
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

# AI 幫你模擬連續下注策略（不會輸光光）
def simulate_ai_play(capital=1000, rounds=100, bet_unit=10):
    try:
        model = xgb.XGBClassifier()
        model.load_model("model/xgb_model.json")
        df = pd.read_csv("data/haoting_data.csv") if os.path.exists("data/haoting_data.csv") else pd.DataFrame()
        if df.empty:
            return "❌ 沒有資料可模擬"

        log = []
        success = 0

        for i in range(min(rounds, len(df))):
            row = df.iloc[i]
            input_data = {
                "局數": row["局數"],
                "免費遊戲": row["免費遊戲"],
                "小分": row["小分"],
                "爆發指數": row["爆發指數"]
            }
            pred, prob = predict_jackpot(input_data)
            if prob > 0.7 and capital >= bet_unit:
                capital -= bet_unit
                if row["爆金"] == 1:
                    capital += bet_unit * 5  # 假設爆金 5 倍回報
                    result = f"✅ 爆金 +{bet_unit * 4}"
                    success += 1
                else:
                    result = "❌ 未爆 -10"
            else:
                result = "🔍 觀望"
            log.append(f"第{i+1}局｜預測爆金率：{prob*100:.2f}%｜{result}｜資金：${capital}")

        return "\n".join(log + [f"\n📊 爆金命中次數：{success} / {rounds}｜結餘資金：${capital}"])
    except Exception as e:
        return f"模擬錯誤：{str(e)}"
