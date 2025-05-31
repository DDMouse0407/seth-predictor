import streamlit as st
import pandas as pd
import datetime
from analyzer import analyze_latest_data, simulate_data, train_xgb_model, predict_jackpot, simulate_ai_play
from scraper_haoting import parse_haoting_page

st.set_page_config(page_title="戰神塞特爆金預測系統", layout="wide")

st.title("🎰 戰神塞特爆金預測系統 v2.0")
st.markdown("開發參考：`haoting.info/nickaa`，自製預測引擎 + 模擬分析 + 爆發建議")

# Sidebar 選單
st.sidebar.header("🔧 模式選擇")
mode = st.sidebar.selectbox("請選擇功能模式：", ["首頁總覽", "模擬分析", "爆發查詢", "AI 模型訓練", "AI 模擬下注", "資料更新（自動爬蟲）"])

# 假資料讀取
def load_history():
    try:
        return pd.read_csv("data/history.csv")
    except:
        return pd.DataFrame(columns=["日期", "局數", "爆金", "小分", "免費遊戲", "爆發指數"])

# 首頁顯示內容
if mode == "首頁總覽":
    st.subheader("📊 系統概況與分析紀錄")
    data = load_history()
    st.metric("總紀錄場次", len(data))
    st.metric("爆金總次數", data['爆金'].sum())
    st.metric("預測平均命中率", f"{(data['爆金'].sum() / len(data) * 100 if len(data)>0 else 0):.2f}%")

    st.markdown("---")
    st.markdown("#### 📈 最近分析紀錄")
    if not data.empty:
        st.dataframe(data.tail(10).sort_values(by="日期", ascending=False))
    else:
        st.info("尚無資料，請先進行模擬分析或手動輸入。")

# 模擬分析頁面
elif mode == "模擬分析":
    st.subheader("🔮 爆金模擬分析器")
    if st.button("執行模擬分析"):
        df = simulate_data()
        st.success("模擬資料產生完成")
        st.dataframe(df.tail(10))

# 爆發查詢頁面
elif mode == "爆發查詢":
    st.subheader("🧠 爆發機率預測查詢")
    with st.form("predict_form"):
        plays = st.number_input("局數", min_value=0, value=50)
        free_game = st.selectbox("免費遊戲是否觸發", [0, 1])
        small_hit = st.selectbox("是否小分", [0, 1])
        burst_index = st.number_input("爆發指數 (可選填)", value=round(plays * 0.05 + small_hit * 10 + free_game * 50, 2))
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
            st.success(f"預測結果：{'💥 爆金' if pred == 1 else '❌ 未爆'}，爆金機率：{prob*100:.2f}%")

# AI 模型訓練頁面
elif mode == "AI 模型訓練":
    st.subheader("🤖 AI 模型訓練模組")
    if st.button("訓練模型（含 haoting 資料）"):
        result = train_xgb_model()
        st.write(result)

# 模擬下注頁面
elif mode == "AI 模擬下注":
    st.subheader("🎮 AI 自動模擬下注")
    st.markdown("AI 根據 haoting 實際資料自動判斷每局是否下注，並計算收益。")
    capital = st.number_input("起始資金", value=1000)
    rounds = st.number_input("模擬局數", value=50, step=10)
    bet_unit = st.number_input("單次下注金額", value=10)
    if st.button("開始模擬"):
        result = simulate_ai_play(capital=int(capital), rounds=int(rounds), bet_unit=int(bet_unit))
        st.text_area("模擬結果", result, height=500)

# 自動爬蟲資料更新
elif mode == "資料更新（自動爬蟲）":
    st.subheader("🌐 從 haoting 網站即時更新資料")
    if st.button("開始抓取最新資料"):
        df = parse_haoting_page()
        if not df.empty:
            st.success(f"成功更新 {len(df)} 筆資料！")
            st.dataframe(df.head())
        else:
            st.warning("抓取失敗或無新資料。")
