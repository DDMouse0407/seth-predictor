import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import train_xgb_model, predict_jackpot, simulate_ai_play
from scraper_haoting import parse_haoting_page

st.set_page_config(page_title="戰神塞特爆金預測系統", layout="wide")
st.title("🎰 戰神塞特爆金預測系統 v2.0")
st.markdown("參考來源：`haoting.info/nickaa`，整合爆金預測 AI + 大數據分析 + 可視化功能")

st.sidebar.header("🔧 功能選擇")
mode = st.sidebar.selectbox("請選擇操作模式：", [
    "首頁總覽", "爆發查詢", "AI 模型訓練", "AI 模擬下注", "資料更新（自動爬蟲）", "爆發趨勢圖表"
])

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/haoting_data.csv")
    except:
        return pd.DataFrame(columns=["日期", "局數", "爆金", "小分", "免費遊戲", "爆發指數"])

if mode == "首頁總覽":
    st.subheader("📊 系統概況與資料統計")
    df = load_data()
    st.metric("總場次", len(df))
    st.metric("爆金次數", df['爆金'].sum())
    st.metric("爆金機率", f"{(df['爆金'].mean()*100 if len(df)>0 else 0):.2f}%")
    st.markdown("#### 🔍 最新紀錄")
    if not df.empty:
        st.dataframe(df.tail(10).sort_values(by="日期", ascending=False))
    else:
        st.info("尚無資料，請先更新或上傳。")

elif mode == "爆發查詢":
    st.subheader("🧠 爆金機率即時預測")
    with st.form("predict_form"):
        plays = st.number_input("局數", min_value=0, value=50)
        free_game = st.selectbox("免費遊戲是否觸發", [0, 1])
        small_hit = st.selectbox("是否為小分", [0, 1])
        burst_index = st.number_input("爆發指數", value=round(plays * 0.05 + small_hit * 10 + free_game * 50, 2))
        submitted = st.form_submit_button("進行預測")
    if submitted:
        input_data = {
            "局數": plays,
            "免費遊戲": free_game,
            "小分": small_hit,
            "爆發指數": burst_index
        }
        pred, prob = predict_jackpot(input_data)
        if pred == -1:
            st.error(prob)
        else:
            st.success(f"預測：{'💥 爆金' if pred else '❌ 未爆'}，機率：{prob*100:.2f}%")

elif mode == "AI 模型訓練":
    st.subheader("🤖 AI 模型訓練器")
    if st.button("重新訓練模型（含最新資料）"):
        result = train_xgb_model()
        st.write(result)

elif mode == "AI 模擬下注":
    st.subheader("🎮 AI 自動下注模擬器")
    capital = st.number_input("初始資金", value=1000)
    rounds = st.number_input("模擬局數", value=50, step=10)
    bet_unit = st.number_input("單注金額", value=10)
    if st.button("開始模擬"):
        result = simulate_ai_play(capital=int(capital), rounds=int(rounds), bet_unit=int(bet_unit))
        st.text_area("模擬結果紀錄：", result, height=500)

elif mode == "資料更新（自動爬蟲）":
    st.subheader("🌐 自動擷取 haoting 最新資料")
    if st.button("開始抓取"):
        df = parse_haoting_page()
        if not df.empty:
            st.success(f"成功擷取 {len(df)} 筆資料")
            st.dataframe(df.head())
        else:
            st.warning("未取得新資料，請稍後再試。")

elif mode == "爆發趨勢圖表":
    st.subheader("📈 爆發指數趨勢可視化")
    df = load_data()
    try:
        df = df.tail(100).reset_index()
        fig, ax1 = plt.subplots()
        ax1.plot(df.index, df['爆發指數'], color='blue')
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['爆金'], color='red', linestyle='dashed')
        ax1.set_xlabel("場次")
        ax1.set_ylabel("爆發指數")
        ax2.set_ylabel("爆金結果")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"繪圖失敗：{e}")
