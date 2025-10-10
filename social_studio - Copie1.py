# social_studio.py
import os, json, time, datetime as dt
from pathlib import Path
from typing import Optional
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DRAFTS_DIR = Path(os.getenv("DRAFTS_DIR", "social_drafts"))
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
PAGE_ID = os.getenv("PAGE_ID", "")
IG_USER_ID = os.getenv("IG_USER_ID", "")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v20.0")

API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

st.set_page_config(page_title="Social Studio – A La Lucarne", layout="wide")

# -------- Helpers ----------
def _read(d: Path, name: str, default: str="") -> str:
    p = d / name
    return p.read_text(encoding="utf-8") if p.exists() else default

def list_drafts():
    return sorted([p for p in DRAFTS_DIR.glob("*") if p.is_dir() and (p/"meta.json").exists()])

def fb_publish_photo(message: str, image_path: Path, schedule_ts: Optional[int] = None):
    url = f"{API_BASE}/{PAGE_ID}/photos"
    files = {"source": open(image_path, "rb")} if image_path.exists() else None
    data = {"access_token": PAGE_ACCESS_TOKEN}
    # Facebook accepte 'caption' pour photos (ou 'message'); on met 'caption'
    data["caption"] = message
    if schedule_ts:
        data["published"] = "false"
        data["scheduled_publish_time"] = str(schedule_ts)
    resp = requests.post(url, data=data, files=files, timeout=60)
    if files: files["source"].close()
    return resp.status_code, resp.text

def fb_publish_link(message: str, link: str, schedule_ts: Optional[int] = None):
    url = f"{API_BASE}/{PAGE_ID}/feed"
    data = {
        "access_token": PAGE_ACCESS_TOKEN,
        "message": message,
        "link": link
    }
    if schedule_ts:
        data["published"] = "false"
        data["scheduled_publish_time"] = str(schedule_ts)
    resp = requests.post(url, data=data, timeout=60)
    return resp.status_code, resp.text

def ig_publish(image_url: str, caption: str):
    # 1) Création du container
    url_media = f"{API_BASE}/{IG_USER_ID}/media"
    data = {
        "image_url": image_url,
        "caption": caption,
        "access_token": PAGE_ACCESS_TOKEN
    }
    r = requests.post(url_media, data=data, timeout=60)
    if r.status_code != 200:
        return r.status_code, f"Create media error: {r.text}"
    creation_id = r.json().get("id")
    # 2) Publication immédiate
    url_pub = f"{API_BASE}/{IG_USER_ID}/media_publish"
    r2 = requests.post(url_pub, data={"creation_id": creation_id, "access_token": PAGE_ACCESS_TOKEN}, timeout=60)
    return r2.status_code, r2.text

def to_unix_ts(date: dt.date, time_: dt.time, tz: dt.tzinfo) -> int:
    dt_local = dt.datetime.combine(date, time_).replace(tzinfo=tz)
    return int(dt_local.timestamp())

# -------- UI ----------
st.title("📣 Social Studio — A La Lucarne")
st.write("Valide, programme et publie tes posts Instagram & Facebook.")

if not PAGE_ACCESS_TOKEN or not PAGE_ID or not IG_USER_ID:
    st.error("Config manquante. Remplis .env : PAGE_ACCESS_TOKEN, PAGE_ID, IG_USER_ID.")
    st.stop()

drafts = list_drafts()
if not drafts:
    st.info("Aucun brouillon. Lance d’abord `python generer_posts_sociaux.py`.")
    st.stop()

sel = st.selectbox("Choisis un brouillon", drafts, format_func=lambda p: p.name)
meta = json.loads((sel/"meta.json").read_text(encoding="utf-8"))

colL, colR = st.columns([1,1])

# ----- Instagram -----
with colL:
    st.subheader("Instagram")
    igA = meta["ig"]["A"]; igB = meta["ig"]["B"]
    tabA, tabB = st.tabs(["Variante A", "Variante B"])
    with tabA:
        st.image(str(sel/igA["image_local"]), caption="Preview IG A", use_container_width=True)
        capA = _read(sel, "ig_caption_A.txt")
        st.text_area("Légende IG A", capA, height=200, key="ig_cap_A")
        if st.button("Publier IG – Variante A", type="primary"):
            status, resp = ig_publish(igA["image_url"], st.session_state["ig_cap_A"])
            st.code(f"IG A → {status}\n{resp}")
    with tabB:
        st.image(str(sel/igB["image_local"]), caption="Preview IG B", use_container_width=True)
        capB = _read(sel, "ig_caption_B.txt")
        st.text_area("Légende IG B", capB, height=200, key="ig_cap_B")
        if st.button("Publier IG – Variante B", type="primary"):
            status, resp = ig_publish(igB["image_url"], st.session_state["ig_cap_B"])
            st.code(f"IG B → {status}\n{resp}")

# ----- Facebook -----
with colR:
    st.subheader("Facebook (Page)")
    fbA = meta["fb"]["A"]; fbB = meta["fb"]["B"]
    mode = st.radio("Type de post FB", ["Photo", "Lien"], horizontal=True)

    tabFA, tabFB = st.tabs(["Variante A", "Variante B"])
    for idx, (tab, fb, cap_file) in enumerate([(tabFA, fbA, "fb_caption_A.txt"), (tabFB, fbB, "fb_caption_B.txt")], start=1):
        with tab:
            img_p = sel / fb["image_local"]
            if img_p.exists():
                st.image(str(img_p), caption=f"Preview FB {'AB'[idx-1]}", use_container_width=True)
            cap = _read(sel, cap_file)
            st.text_area("Message FB", cap, height=200, key=f"fb_cap_{idx}")
            schedule = st.checkbox("Programmer", key=f"fb_sched_{idx}")
            sched_ts = None
            if schedule:
                d = st.date_input("Date de publication", value=dt.date.today()+dt.timedelta(days=1), key=f"date_{idx}")
                t = st.time_input("Heure (locale)", value=dt.time(9, 0), key=f"time_{idx}")
                sched_ts = int(dt.datetime.combine(d, t).timestamp())

            if st.button(f"Publier FB – Variante {'AB'[idx-1]}", type="primary", key=f"fb_pub_{idx}"):
                if mode == "Photo":
                    status, resp = fb_publish_photo(st.session_state[f"fb_cap_{idx}"], img_p, schedule_ts=sched_ts)
                else:
                    link = meta.get("video_url") or ""
                    status, resp = fb_publish_link(st.session_state[f"fb_cap_{idx}"], link, schedule_ts=sched_ts)
                st.code(f"FB {'AB'[idx-1]} → {status}\n{resp}")

st.success("Prêt. Tu peux publier/régler par brouillon, ou changer de brouillon ci-dessus.")
