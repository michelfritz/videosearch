# social_studio.py — Interface de validation / publication IG & FB
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

# -------- Helpers de fichiers ----------
def _read(d: Path, name: str, default: str="") -> str:
    p = d / name
    return p.read_text(encoding="utf-8") if p.exists() else default

def _resolve_local(base_dir: Path, rel: str) -> Optional[Path]:
    """Résout de façon robuste un chemin 'image_local' provenant des meta.json."""
    if not rel:
        return None
    # 1) Chemin relatif direct (ex: images/IG_A.jpg)
    p = base_dir / rel
    if p.exists():
        return p
    # 2) Dans le sous-dossier images/ (si rel était juste 'IG_A.jpg')
    fn = Path(rel).name
    p2 = base_dir / "images" / fn
    if p2.exists():
        return p2
    # 3) À la racine du brouillon
    p3 = base_dir / fn
    if p3.exists():
        return p3
    return None

def list_drafts():
    return sorted([p for p in DRAFTS_DIR.glob("*") if p.is_dir() and (p/"meta.json").exists()])

# --------- Facebook API ----------
def fb_publish_photo(message: str, image_path: Path, schedule_ts: Optional[int] = None):
    url = f"{API_BASE}/{PAGE_ID}/photos"
    files = {"source": open(image_path, "rb")} if (image_path and image_path.exists()) else None
    data = {"access_token": PAGE_ACCESS_TOKEN}
    data["caption"] = message or ""
    if schedule_ts:
        data["published"] = "false"
        data["scheduled_publish_time"] = str(schedule_ts)
    resp = requests.post(url, data=data, files=files, timeout=60)
    if files:
        try:
            files["source"].close()
        except Exception:
            pass
    return resp.status_code, resp.text

def fb_publish_link(message: str, link: str, schedule_ts: Optional[int] = None):
    url = f"{API_BASE}/{PAGE_ID}/feed"
    data = {
        "access_token": PAGE_ACCESS_TOKEN,
        "message": message or "",
        "link": link or ""
    }
    if schedule_ts:
        data["published"] = "false"
        data["scheduled_publish_time"] = str(schedule_ts)
    resp = requests.post(url, data=data, timeout=60)
    return resp.status_code, resp.text

# --------- Instagram API ----------
def ig_publish(image_url: str, caption: str):
    # 1) Création du container
    url_media = f"{API_BASE}/{IG_USER_ID}/media"
    data = {
        "image_url": image_url,
        "caption": caption or "",
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
        img_p = _resolve_local(sel, igA.get("image_local",""))
        if img_p and img_p.exists():
            st.image(str(img_p), caption="Preview IG A", use_container_width=True)
        elif igA.get("image_url"):
            st.image(igA["image_url"], caption="Preview IG A (URL)", use_container_width=True)
        else:
            st.warning("Image IG A introuvable.")
        capA = _read(sel, igA.get("caption_file","ig_caption_A.txt"))
        st.text_area("Légende IG A", capA, height=200, key="ig_cap_A")
        if st.button("Publier IG – Variante A", type="primary"):
            if igA.get("image_url"):
                status, resp = ig_publish(igA["image_url"], st.session_state.get("ig_cap_A",""))
                st.code(f"IG A → {status}\n{resp}")
            else:
                st.error("Pas d'URL publique pour l'image IG A.")
    with tabB:
        img_p = _resolve_local(sel, igB.get("image_local",""))
        if img_p and img_p.exists():
            st.image(str(img_p), caption="Preview IG B", use_container_width=True)
        elif igB.get("image_url"):
            st.image(igB["image_url"], caption="Preview IG B (URL)", use_container_width=True)
        else:
            st.warning("Image IG B introuvable.")
        capB = _read(sel, igB.get("caption_file","ig_caption_B.txt"))
        st.text_area("Légende IG B", capB, height=200, key="ig_cap_B")
        if st.button("Publier IG – Variante B", type="primary"):
            if igB.get("image_url"):
                status, resp = ig_publish(igB["image_url"], st.session_state.get("ig_cap_B",""))
                st.code(f"IG B → {status}\n{resp}")
            else:
                st.error("Pas d'URL publique pour l'image IG B.")

# ----- Facebook -----
with colR:
    st.subheader("Facebook (Page)")
    fbA = meta["fb"]["A"]; fbB = meta["fb"]["B"]
    mode = st.radio("Type de post FB", ["Photo", "Lien"], horizontal=True)

    tabFA, tabFB = st.tabs(["Variante A", "Variante B"])

    def fb_block(tab, fb_dict, cap_file_key: str, key_prefix: str):
        with tab:
            img_p = _resolve_local(sel, fb_dict.get("image_local",""))
            if img_p and img_p.exists():
                st.image(str(img_p), caption=f"Preview FB {key_prefix}", use_container_width=True)
            else:
                st.warning("Image FB introuvable.")
            cap = _read(sel, fb_dict.get("caption_file", cap_file_key))
            st.text_area("Message FB", cap, height=200, key=f"fb_cap_{key_prefix}")
            schedule = st.checkbox("Programmer", key=f"fb_sched_{key_prefix}")
            sched_ts = None
            if schedule:
                d = st.date_input("Date de publication", value=dt.date.today()+dt.timedelta(days=1), key=f"date_{key_prefix}")
                t = st.time_input("Heure (locale)", value=dt.time(9, 0), key=f"time_{key_prefix}")
                sched_ts = int(dt.datetime.combine(d, t).timestamp())

            if st.button(f"Publier FB – Variante {key_prefix}", type="primary", key=f"fb_pub_{key_prefix}"):
                if mode == "Photo":
                    if img_p and img_p.exists():
                        status, resp = fb_publish_photo(st.session_state.get(f"fb_cap_{key_prefix}",""), img_p, schedule_ts=sched_ts)
                        st.code(f"FB {key_prefix} → {status}\n{resp}")
                    else:
                        st.error("Image manquante pour publication Photo.")
                else:
                    link = meta.get("video_url") or ""
                    status, resp = fb_publish_link(st.session_state.get(f"fb_cap_{key_prefix}",""), link, schedule_ts=sched_ts)
                    st.code(f"FB {key_prefix} → {status}\n{resp}")

    fb_block(tabFA, fbA, "fb_caption_A.txt", "A")
    fb_block(tabFB, fbB, "fb_caption_B.txt", "B")

st.success("Prêt. Tu peux publier/régler par brouillon, ou changer de brouillon ci-dessus.")
