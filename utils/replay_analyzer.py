import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import joblib
import numpy as np
import re
import os

from replay_utils import analyze_replay_url

st.set_page_config(page_title="賽特分析系統 - 自動序號工具", layout="centered")
st.title("🔑 賽特序號自動分析工具 v2.6")
st.markdown("請輸入每日序號與帳號資訊，系統將自動送出分析請求、擷取爆金圖片與影片回放網址，結合 AI 預測與下注建議，並持續記錄強化學習。")

@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_burst_predictor.pkl")
    except:
        return None

def save_daily_training_data(record):
    file = "daily_training_log.csv"
    df = pd.DataFrame([record])
    if os.path.exists(file):
        df.to_csv(file, mode="a", header=False, index=False)
    else:
        df.to_csv(file, index=False)

def make_betting_decision(prob, threshold=0.6):
    if prob >= threshold:
        return f"✅ 建議下注（信心值 {prob*100:.1f}%）"
    else:
        return f"⛔ 建議觀望（信心值 {prob*100:.1f}%）"

model = load_model()

with st.form("serial_form"):
    serial = st.text_input("今日序號", placeholder="例如：115511")
    account = st.text_input("會員帳號", placeholder="例如：moneymm258")
    amount = st.text_input("設定金額", value="1000")
    table = st.text_input("桌號", value="109")
    device = st.selectbox("裝置類型", ["ios", "android", "pc"])
    game = st.selectbox("選擇遊戲", ["ATG-賽特", "ATG-其他"])
    submitted = st.form_submit_button("🚀 開始分析")

if submitted:
    with st.spinner("正在提交序號並擷取分析結果..."):
        payload = {
            "serial": serial,
            "device": device,
            "account": account,
            "amount": amount,
            "game": game,
            "table": table
        }
        try:
            res = requests.post("https://haoting.info/verifySerial.php", data=payload)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                icons = soup.find_all("img")
                links = soup.find_all("a", href=True)

                results = []
                level_map = {
                    "無爆發": 0,
                    "Big Win": 1,
                    "Super Win": 2,
                    "Mega Win": 3,
                    "Ultra Win": 4,
                    "Legendary Win": 5,
                }

                for icon in icons:
                    src = icon.get("src")
                    if src:
                        if "legendary" in src.lower():
                            win_type = "Legendary Win"
                        elif "ultra" in src.lower():
                            win_type = "Ultra Win"
                        elif "mega" in src.lower():
                            win_type = "Mega Win"
                        elif "super" in src.lower():
                            win_type = "Super Win"
                        elif "big" in src.lower():
                            win_type = "Big Win"
                        else:
                            win_type = "無爆發"

                        results.append({
                            "時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "爆發等級": win_type,
                            "圖片": src,
                            "數值": level_map.get(win_type, 0)
                        })

                replay_urls = []
                for a in links:
                    href = a["href"]
                    if "godeebxp.com/egames" in href and "egyptian-mythology" in href:
                        replay_urls.append(href)

                if results:
                    df = pd.DataFrame(results)
                    st.success("🎉 爆金資料擷取成功！")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 下載結果 CSV", data=csv, file_name="haoting_data.csv", mime="text/csv")

                    if model:
                        X_input = np.array(df["數值"].tail(5)).reshape(1, -1)
                        if X_input.shape[1] < 5:
                            X_input = np.pad(X_input, ((0,0),(5-X_input.shape[1],0)))
                        pred = model.predict_proba(X_input)[0][1]
                        decision = make_betting_decision(pred)

                        st.markdown(f"### 🤖 AI 預測下一局爆金機率：**{pred*100:.2f}%**")
                        st.markdown(f"### 💰 自動下注建議：{decision}")

                        save_daily_training_data({
                            "日期時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "序號": serial,
                            "帳號": account,
                            "爆發分數陣列": df["數值"].tolist(),
                            "爆金機率": round(pred, 4),
                            "下注建議": decision
                        })
                    else:
                        st.warning("尚未載入 AI 模型，請確認 xgb_burst_predictor.pkl 存在於目錄中。")
                else:
                    st.warning("未偵測到爆金資訊圖片，可能本次無爆發等級資料。")

                if replay_urls:
                    st.markdown("---")
                    st.markdown("### 🎞️ 偵測到回放網址：")
                    for url in replay_urls:
                        label = analyze_replay_url(url)
                        st.write(f"{url} 👉 分析結果：**{label}**")

            else:
                st.error(f"分析失敗，狀態碼：{res.status_code}")
        except Exception as e:
            st.error(f"發生錯誤：{e}")
