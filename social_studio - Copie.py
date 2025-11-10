# social_studio.py — Interface de validation / publication IG & FB (base: version prod)
# - Carrousel d’images par variante (⟵ ⟶) sur candidates[]
# - Édition / validation des légendes (sauvegarde .txt + selected dans meta.json)
# - Vérifs IG: URL publique requise

import os, json, time, datetime as dt
from pathlib import Path
from typing import Optional, Dict, List
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
st.title("📣 Social Studio — A La Lucarne")
st.write("Valide, retouche et publie tes posts Instagram & Facebook.")

# -------- Helpers ----------
def _read(p: Path, default=""):
    try: return p.read_text(encoding="utf-8")
    except Exception: return default

def _save(p: Path, text: str):
    p.write_text(text, encoding="utf-8")

def _resolve_local(base_dir: Path, rel: str) -> Optional[Path]:
    if not rel: return None
    p = base_dir / rel
    if p.exists(): return p
    fn = Path(rel).name
    p2 = base_dir / "images" / fn
    if p2.exists(): return p2
    p3 = base_dir / fn
    if p3.exists(): return p3
    return None

def _load_meta(root: Path) -> Optional[dict]:
    m = root/"meta.json"
    if not m.exists(): return None
    try: return json.loads(m.read_text(encoding="utf-8"))
    except Exception: return None

def _save_meta(root: Path, meta: dict):
    (root/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def list_drafts():
    return sorted([p for p in DRAFTS_DIR.glob("*") if p.is_dir() and (p/"meta.json").exists()])

# --------- Facebook API ----------
def fb_publish_photo(message: str, image_path: Path, schedule_ts: Optional[int] = None):
    url = f"{API_BASE}/{PAGE_ID}/photos"
    files = {"source": open(image_path, "rb")} if (image_path and image_path.exists()) else None
    data = {"access_token": PAGE_ACCESS_TOKEN, "caption": message or ""}
    if schedule_ts:
        data["published"] = "false"
        data["scheduled_publish_time"] = str(schedule_ts)
    r = requests.post(url, data=data, files=files, timeout=60)
    if files:
        try: files["source"].close()
        except Exception: pass
    return r.status_code, r.text

def fb_publish_link(message: str, link: str, schedule_ts: Optional[int] = None):
    url = f"{API_BASE}/{PAGE_ID}/feed"
    data = {"access_token": PAGE_ACCESS_TOKEN, "message": message or "", "link": link or ""}
    if schedule_ts:
        data["published"] = "false"
        data["scheduled_publish_time"] = str(schedule_ts)
    r = requests.post(url, data=data, timeout=60)
    return r.status_code, r.text

# --------- Instagram API ----------
def ig_publish(image_url: str, caption: str):
    url_media = f"{API_BASE}/{IG_USER_ID}/media"
    data = {"image_url": image_url, "caption": caption or "", "access_token": PAGE_ACCESS_TOKEN}
    r = requests.post(url_media, data=data, timeout=60)
    if r.status_code != 200:
        return r.status_code, f"Create media error: {r.text}"
    creation_id = r.json().get("id")
    url_pub = f"{API_BASE}/{IG_USER_ID}/media_publish"
    r2 = requests.post(url_pub, data={"creation_id": creation_id, "access_token": PAGE_ACCESS_TOKEN}, timeout=60)
    return r2.status_code, r2.text

# -------- UI : sélection du brouillon ----------
if not PAGE_ACCESS_TOKEN or not PAGE_ID or not IG_USER_ID:
    st.error("Config manquante. Remplis .env : PAGE_ACCESS_TOKEN, PAGE_ID, IG_USER_ID.")
    st.stop()

drafts = list_drafts()
if not drafts:
    st.info("Aucun brouillon. Lance d’abord `python generer_posts_sociaux.py`.")
    st.stop()

sel = st.selectbox("Choisis un brouillon", drafts, format_func=lambda p: p.name)
meta = _load_meta(sel)
if not meta:
    st.error("meta.json introuvable.")
    st.stop()

st.header(meta.get("title","(Sans titre)"))
st.caption(f"🎬 Fichier: {meta.get('video_file')} | 🔗 URL: {meta.get('video_url','')}")

# -------- UI helpers : carrousel + édition/validation ----------
def render_variant(area: str, node: dict, cap_file: str, is_instagram: bool, state_key: str):
    st.markdown(f"### {area}")
    cands: List[Dict] = node.get("candidates", [])
    if not cands:
        st.warning("Aucun candidat image pour cette variante.")
        return

    sel_idx = int(node.get("selected", 0) or 0)
    if state_key not in st.session_state:
        st.session_state[state_key] = sel_idx
    idx = st.session_state[state_key] % len(cands)
    cand = cands[idx]

    cols = st.columns([1,2,1])
    with cols[0]:
        if st.button("⟵", key=f"{area}_prev"):
            st.session_state[state_key] = (idx - 1) % len(cands)
            idx = st.session_state[state_key]; cand = cands[idx]
    with cols[1]:
        img_local = _resolve_local(sel, cand.get("image_local",""))
        if img_local:
            st.image(str(img_local), use_container_width=True)
        elif cand.get("image_url"):
            st.image(cand["image_url"], use_container_width=True)
        else:
            st.error("Image introuvable (ni locale, ni URL).")
    with cols[2]:
        if st.button("⟶", key=f"{area}_next"):
            st.session_state[state_key] = (idx + 1) % len(cands)
            idx = st.session_state[state_key]; cand = cands[idx]

    st.caption(f"Source: **{cand.get('source','?')}** — URL publique: {'✅' if cand.get('image_url') else '❌'}")
    st.progress((idx+1)/len(cands), text=f"{idx+1} / {len(cands)}")

    # Légende : édition / validation
    cap_path = sel / cap_file
    edit_key = f"{area}_edit_mode"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = False

    if st.session_state[edit_key]:
        txt = st.text_area("✏️ Modifier la légende", _read(cap_path), height=220, key=f"{area}_editor")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Valider", key=f"{area}_validate"):
                _save(cap_path, txt)
                node["selected"] = idx
                _save_meta(sel, meta)
                st.session_state[edit_key] = False
                st.success("Légende enregistrée et image sélectionnée.")
        with c2:
            if st.button("↩️ Annuler", key=f"{area}_cancel"):
                st.session_state[edit_key] = False
    else:
        st.text_area("Légende", _read(cap_path), height=180, disabled=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✏️ Éditer", key=f"{area}_edit"):
                st.session_state[edit_key] = True
        with c2:
            if st.button("💾 Sauver sélection", key=f"{area}_save"):
                node["selected"] = idx
                _save_meta(sel, meta)
                st.success("Sélection d’image sauvegardée.")
        with c3:
            label = "📤 Publier IG" if is_instagram else "📣 Publier FB"
            if st.button(label, key=f"{area}_publish"):
                message = _read(cap_path)
                if is_instagram:
                    if not cand.get("image_url"):
                        st.error("Instagram requiert une URL publique. Choisis un candidat Unsplash (URL ✅) ou héberge l’image.")
                    else:
                        status, resp = ig_publish(cand["image_url"], message)
                        st.code(f"IG → {status}\n{resp}")
                else:
                    # Choix du mode (Photo / Lien)
                    st.info("FB : publication Photo (image locale si dispo) ou Lien (URL vidéo).")
                    mode = st.radio("Mode de publication FB", ["Photo", "Lien"], horizontal=True, key=f"{area}_fbmode")
                    sched = st.checkbox("Programmer", key=f"{area}_sched")
                    sched_ts = None
                    if sched:
                        d = st.date_input("Date", value=dt.date.today()+dt.timedelta(days=1), key=f"{area}_date")
                        t = st.time_input("Heure", value=dt.time(9,0), key=f"{area}_time")
                        sched_ts = int(dt.datetime.combine(d,t).timestamp())
                    if mode == "Photo":
                        img_local = _resolve_local(sel, cand.get("image_local",""))
                        if img_local:
                            status, resp = fb_publish_photo(message, img_local, schedule_ts=sched_ts)
                            st.code(f"FB (photo) → {status}\n{resp}")
                        else:
                            st.error("Image locale manquante pour le mode Photo.")
                    else:
                        link = meta.get("video_url") or ""
                        status, resp = fb_publish_link(message, link, schedule_ts=sched_ts)
                        st.code(f"FB (lien) → {status}\n{resp}")

# -------- Grille IG / FB ----------
colL, colR = st.columns(2)
with colL:
    st.subheader("Instagram")
    render_variant("Instagram — Variante A", meta["ig"]["A"], meta["ig"]["A"]["caption_file"], True, state_key="igA_idx")
    st.markdown("---")
    render_variant("Instagram — Variante B", meta["ig"]["B"], meta["ig"]["B"]["caption_file"], True, state_key="igB_idx")
with colR:
    st.subheader("Facebook (Page)")
    render_variant("Facebook — Variante A", meta["fb"]["A"], meta["fb"]["A"]["caption_file"], False, state_key="fbA_idx")
    st.markdown("---")
    render_variant("Facebook — Variante B", meta["fb"]["B"], meta["fb"]["B"]["caption_file"], False, state_key="fbB_idx")

st.success("Prêt. Utilise ⟵ / ⟶ pour choisir l’image, ✏️ pour éditer, puis publie.")
