
def analyze_replay_url(url: str) -> str:
    if "legendary" in url.lower():
        return "Legendary Win"
    elif "ultra" in url.lower():
        return "Ultra Win"
    elif "mega" in url.lower():
        return "Mega Win"
    elif "super" in url.lower():
        return "Super Win"
    elif "big" in url.lower():
        return "Big Win"
    else:
        return "無爆發"
