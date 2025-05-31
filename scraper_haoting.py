import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def parse_haoting_page(url="https://ww.haoting.info/nickaa"):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr")[1:]
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5: continue
            date = datetime.now().strftime("%Y-%m-%d")
            plays = int(cols[1].text.strip())
            small = 1 if "小分" in cols[2].text else 0
            free = 1 if "免費" in cols[2].text else 0
            jackpot = 1 if "爆" in cols[3].text else 0
            burst_index = round(plays * 0.05 + small * 10 + free * 50, 2)
            data.append([date, plays, jackpot, small, free, burst_index])
        df = pd.DataFrame(data, columns=["日期", "局數", "爆金", "小分", "免費遊戲", "爆發指數"])
        df.to_csv("data/haoting_data.csv", index=False)
        return df
    except Exception as e:
        print(f"[錯誤] {e}")
        return pd.DataFrame()
