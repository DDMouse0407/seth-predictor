import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import joblib
import numpy as np

st.set_page_config(page_title="賽特分析系統 - 自動序號工具", layout="centered")
st.title("🔑 賽特序號自動分析工具 v2.0")
st.markdown("請輸入每日序號與帳號資訊，系統將自動送出分析請求並擷取爆金資料與預測下一局爆發機率。")

@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_burst_predictor.pkl")
    except:
        return None

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
                        st.markdown(f"### 🤖 AI 預測下一局爆金機率：**{pred*100:.2f}%**")
                    else:
                        st.warning("尚未載入 AI 模型，請確認 xgb_burst_predictor.pkl 存在於目錄中。")
                else:
                    st.warning("未偵測到爆金資訊圖片，可能本次無爆發等級資料。")
            else:
                st.error(f"分析失敗，狀態碼：{res.status_code}")
        except Exception as e:
            st.error(f"發生錯誤：{e}")
