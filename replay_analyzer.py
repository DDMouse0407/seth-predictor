import re

# 自訂爆發等級關鍵字與對應標籤
def analyze_replay_url(url: str) -> str:
    """
    分析給定的回放網址，判斷其可能的爆發等級
    
    假設使用 URL 中常見的特徵關鍵字進行簡易比對分析
    
    :param url: 回放網址
    :return: 分析結果（Legendary / Ultra / Mega / Super / Big / 無爆發）
    """
    lower_url = url.lower()

    if "legendary" in lower_url:
        return "Legendary Win"
    elif "ultra" in lower_url:
        return "Ultra Win"
    elif "mega" in lower_url:
        return "Mega Win"
    elif "super" in lower_url:
        return "Super Win"
    elif "big" in lower_url:
        return "Big Win"
    else:
        return "無爆發"


# 可擴充：日後加上影片特效辨識或 ts、spinId 特徵辨識也可在此擴展
