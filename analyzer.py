import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

MODEL_PATH = "models/xgb_model.pkl"
DATA_PATH = "data/haoting_data.csv"

def train_xgb_model():
    if not os.path.exists(DATA_PATH):
        return "❌ 無可用資料訓練模型"

    df = pd.read_csv(DATA_PATH)
    if df.empty or '爆金' not in df.columns:
        return "❌ 資料格式錯誤或為空"

    X = df[['局數', '免費遊戲', '小分', '爆發指數']]
    y = df['爆金']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 儲存模型
    model.save_model(MODEL_PATH)

    acc = accuracy_score(y_test, model.predict(X_test))
    return f"✅ 模型訓練完成，測試集準確率：{acc:.2%}"

def predict_jackpot(input_data):
    if not os.path.exists(MODEL_PATH):
        return -1, "❌ 尚未訓練模型"

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob > 0.5)

    return pred, prob

def simulate_ai_play(capital=1000, rounds=50, bet_unit=10):
    try:
        df = pd.read_csv(DATA_PATH).tail(rounds)
    except:
        return "❌ 模擬失敗，請確認資料是否存在"

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    logs = []
    balance = capital
    hit_count = 0

    for i, row in df.iterrows():
        input_df = pd.DataFrame([{
            '局數': row['局數'],
            '免費遊戲': row['免費遊戲'],
            '小分': row['小分'],
            '爆發指數': row['爆發指數']
        }])
        prob = model.predict_proba(input_df)[0][1]
        should_bet = prob > 0.7

        if should_bet:
            if row['爆金'] == 1:
                balance += bet_unit * 4  # 假設爆金回報是 +4 倍
                hit_count += 1
                logs.append(f"第{i+1}局｜爆金率：{prob:.2%}｜✅ 爆金｜資金：${balance}")
            else:
                balance -= bet_unit
                logs.append(f"第{i+1}局｜爆金率：{prob:.2%}｜❌ 未爆｜資金：${balance}")
        else:
            logs.append(f"第{i+1}局｜爆金率：{prob:.2%}｜🔍 觀望｜資金：${balance}")

    summary = f"📊 命中次數：{hit_count}／{rounds} 局｜最終結餘：${balance}"
    return "\n".join(logs + ["", summary])
