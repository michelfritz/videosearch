# dedupe_urls.py
import pandas as pd, unicodedata, re, os

def norm_strict(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u2000-\u200B\u202F\u205F\u3000]", " ", s)
    s = re.sub(r"[\-‐‑‒–—−]+", "-", s)
    s = re.sub(r"[^a-z0-9\-\._ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "." in s: s = s.rsplit(".", 1)[0]
    return s

df = pd.read_csv("urls.csv", encoding="utf-8")
df["key_strict"] = df["fichier"].map(norm_strict)
df["prio"] = (df["url"].fillna("").astype(str).str.strip() != "").astype(int)
df = df.sort_values(["key_strict","prio"], ascending=[True, False])
df = df.drop_duplicates(subset=["key_strict"], keep="first").drop(columns=["prio"])
df.to_csv("urls.csv", index=False, encoding="utf-8")
print("[OK] urls.csv dé‑dupliqué")
