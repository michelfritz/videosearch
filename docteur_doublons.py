# doctor_blocs.py
from pathlib import Path
import unicodedata, re, os

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

by_key = {}
for p in Path("blocs").glob("*.csv"):
    k = norm_strict(p.stem)
    by_key.setdefault(k, []).append(p.name)

print("== Doublons détectés (même clé stricte) ==")
for k, files in by_key.items():
    if len(files) > 1:
        print(f" - {k}: {files}")
