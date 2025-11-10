
from openai import OpenAI
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
import chardet
import re, html, urllib.parse, mimetypes, base64
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# =========================
#   MODES & CONSTANTES
# =========================
FORCE_NUMPY = True   # on force la voie numpy (désactive FAISS pour éviter erreurs locales)
DEBUG_MODE  = True   # traces visibles dans l'UI

# =========================
#   CHEMINS ROBUSTES
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = BASE_DIR
NEWSLETTER_DIR = os.path.join(BASE_DIR, "newsletters")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_transcripts")

# =========================
#   OPENAI KEY
# =========================
def get_openai_key():
    key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not key:
        st.warning("Clé OpenAI absente : l'IA sera désactivée (l'app continue).")
        return None
    return key

# Définit la clé aussi pour certaines libs (legacy)
openai.api_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)

# =========================
#   PAGE SETUP
# =========================
st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")
st.caption(f"Clé OpenAI détectée : {bool(get_openai_key())}")

# =========================
#   HELPERS GÉNÉRAUX
# =========================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("themes"):
            rename_map[c] = "themes"
        elif lc == "fichier":
            rename_map[c] = "fichier"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _split_and_clean_tags(value) -> list[str]:
    if value is None: return []
    s = str(value)
    for sep in [" ; ", ";", ",", " / ", "/", "\\", "  "]:
        s = s.replace(sep, "|")
    parts = [p.strip() for p in s.split("|")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def init_state():
    defaults = {
        "nav": "🔍 Recherche",
        "search_query": "",
        "video_search": "",
        "selected_theme": "",
        "reset_search": False,
        "selected_video": None,
        "user_question": "",
        "show_thumbs": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

def show_image(img, width=None, caption=None):
    if not img:
        st.write("🖼️ Miniature indisponible")
        return
    try:
        if width is None or int(width) <= 0:
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.image(img, caption=caption, width=int(width))
    except Exception:
        st.image(img, caption=caption, use_container_width=True)

def to_str(x, default=""):
    if x is None: return default
    try:
        if pd.isna(x): return default
    except Exception:
        pass
    return str(x)

def extract_youtube_id(url) -> str:
    s = to_str(url, "").strip()
    if not s: return ""
    if "watch?v=" in s:
        part = s.split("watch?v=")[-1]
        return part.split("&")[0]
    if "youtu.be/" in s:
        part = s.split("youtu.be/")[-1]
        return part.split("?")[0]
    return ""

def to_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return default
        return int(float(x))
    except Exception:
        return default

# =========================
#  TAGS SCROLLER
# =========================
def render_tags_scroller(themes_str: str, uid: str, height: int = 120):
    if not themes_str: return
    chips = [t.strip() for t in str(themes_str).split("|") if t.strip()]
    if not chips: return
    items_html = "".join([f"<span class='tag'>{html.escape(c)}</span>" for c in chips])
    html_block = f"""
    <style>
      .tagbox-{uid} .bar{{display:flex;align-items:center;gap:.25rem;margin:.25rem 0;}}
      .tagbox-{uid} .wrap{{display:grid;grid-auto-flow:column;grid-template-rows:repeat(2,auto);
                           gap:8px 8px;overflow-x:auto;overflow-y:hidden;padding:6px 6px;
                           scroll-behavior:smooth; overscroll-behavior:contain;
                           -webkit-overflow-scrolling:touch; scrollbar-width:thin;}}
      .tagbox-{uid} .wrap::-webkit-scrollbar{{height:8px}}
      .tagbox-{uid} .wrap::-webkit-scrollbar-thumb{{background:rgba(0,0,0,.15);border-radius:8px}}
      .tagbox-{uid} .tag{{display:inline-flex;align-items:center;justify-content:center;text-align:center;
                          white-space:nowrap;background:#D0E8FF;color:#0A2540;
                          border-radius:999px;padding:6px 14px;font-size:13px;border:1px solid rgba(0,0,0,.05);}}
      .tagbox-{uid} .btn{{border:0;background:transparent;color:#6b7280;font-size:22px;cursor:pointer;
                          padding:0 6px;line-height:1}}
      .tagbox-{uid} .btn:focus{{outline:none}}
    </style>
    <div class="tagbox-{uid}">
      <div class="bar">
        <button class="btn" onclick="document.getElementById('wrap-{uid}').scrollBy({{left:-320,behavior:'smooth'}})">&#9664;</button>
        <div id="wrap-{uid}" class="wrap" style="height:{height-24}px">{items_html}</div>
        <button class="btn" onclick="document.getElementById('wrap-{uid}').scrollBy({{left:320,behavior:'smooth'}})">&#9654;</button>
      </div>
    </div>
    """
    st.components.v1.html(html_block, height=height, scrolling=False)

def render_tags_scroller_interactive(tags: list[str], uid: str, height: int = 136):
    if not tags: return
    items_html = "".join(
        f"<a class='tag' href='?select_tag={urllib.parse.quote(t)}' role='button'>{html.escape(t)}</a>"
        for t in tags
    )
    html_block = f"""
    <style>
      .sbox-{uid} .bar{{display:flex;align-items:center;gap:.25rem;margin:.25rem 0;}}
      .sbox-{uid} .wrap{{
         display:grid;grid-auto-flow:column;grid-template-rows:repeat(2,auto);
         gap:8px 8px;overflow-x:auto;overflow-y:hidden;padding:6px 6px;
         scroll-behavior:smooth;-webkit-overflow-scrolling:touch;scrollbar-width:thin;
      }}
      .sbox-{uid} .wrap::-webkit-scrollbar{{height:8px}}
      .sbox-{uid} .wrap::-webkit-scrollbar-thumb{{background:rgba(0,0,0,.15);border-radius:8px}}
      .sbox-{uid} .tag{{
         display:inline-flex;align-items:center;justify-content:center;text-align:center;white-space:nowrap;
         background:#D0E8FF;color:#0A2540;border-radius:999px;padding:6px 14px;font-size:13px;
         border:1px solid rgba(0,0,0,.05);text-decoration:none
      }}
      .sbox-{uid} .btn{{border:0;background:transparent;color:#6b7280;font-size:22px;cursor:pointer;padding:0 6px;line-height:1}}
      .sbox-{uid} .btn:focus{{outline:none}}
    </style>
    <div class="sbox-{uid}">
      <div class="bar">
        <button class="btn" onclick="document.getElementById('swrap-{uid}').scrollBy({{left:-320,behavior:'smooth'}})">&#9664;</button>
        <div id="swrap-{uid}" class="wrap" style="height:{height-24}px">{items_html}</div>
        <button class="btn" onclick="document.getElementById('swrap-{uid}').scrollBy({{left:320,behavior:'smooth'}})">&#9654;</button>
      </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)

def handle_select_tag_from_query():
    try:
        qp = st.query_params
        tag = qp.get("select_tag", None)
        if isinstance(tag, list): tag = tag[0] if tag else None
        if tag:
            st.session_state.selected_theme = tag
            st.session_state.nav = "🔍 Recherche"
            try:
                if hasattr(st, "query_params"):
                    all_q = dict(st.query_params)
                    all_q.pop("select_tag", None)
                    st.query_params = all_q
            except Exception:
                pass
            st.session_state.reset_search = True
    except Exception:
        pass

init_state()
handle_select_tag_from_query()

# Logo + Titre
show_image("logo_lucarne.png", width=180)
st.markdown("# 📚 Base de connaissance A LA LUCARNE")

# =========================
#   DATA LOADING HELPERS
# =========================
def detect_encoding(file_path):
    try:
        with open(os.path.join(DATA_DIR, file_path), 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result.get('encoding') or "utf-8"
    except Exception:
        return "utf-8"

def safe_read_csv(path_like, **kw):
    full = os.path.join(DATA_DIR, path_like)
    enc = kw.pop("encoding", None) or detect_encoding(path_like)
    try:
        return pd.read_csv(full, encoding=enc, **kw)
    except FileNotFoundError:
        st.warning(f"Fichier introuvable : {path_like}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lecture CSV ({path_like}) : {e}")
        return pd.DataFrame()

def safe_load_pickle(path_like):
    full = os.path.join(DATA_DIR, path_like)
    try:
        with open(full, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"Fichier introuvable : {path_like}")
        return None
    except Exception as e:
        st.error(f"Erreur lecture pickle ({path_like}) : {e}")
        return None

def charger_newsletter_html(nom_fichier):
    chemin = os.path.join(NEWSLETTER_DIR, f"{nom_fichier}.html")
    if os.path.exists(chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            st.error(f"Impossible de lire {chemin} : {e}")
            return None
    return None

def _inline_local_images(html_doc: str, base_folder: str) -> str:
    def repl_src(m):
        quote = m.group(1)
        src   = m.group(2).strip()
        if not src or src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            return m.group(0)
        candidate = src
        if candidate.startswith("/"):
            abs_path = os.path.join(BASE_DIR, candidate.lstrip("/"))
        elif candidate.startswith("newsletters/"):
            abs_path = os.path.join(BASE_DIR, candidate)
        else:
            abs_path = os.path.join(base_folder, candidate)
        if not os.path.exists(abs_path):
            return m.group(0)
        try:
            mime, _ = mimetypes.guess_type(abs_path)
            if not mime:
                mime = "application/octet-stream"
            with open(abs_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f'src={quote}data:{mime};base64,{b64}{quote}'
        except Exception:
            return m.group(0)

    html_doc = re.sub(r'src=(")([^"]+)\1', repl_src, html_doc, flags=re.IGNORECASE)
    html_doc = re.sub(r"src=(')([^']+)\1", repl_src, html_doc, flags=re.IGNORECASE)
    return html_doc

def fix_newsletter_html(html_src: str, base_folder: str = NEWSLETTER_DIR) -> str:
    html_doc = html_src or ""
    if not html_doc: return ""

    html_doc = html_doc.replace('src="images/', f'src="{os.path.join(base_folder, "images")}/')
    html_doc = html_doc.replace("src='images/", f"src='{os.path.join(base_folder, 'images')}/")
    html_doc = html_doc.replace('href="images/', f'href="{os.path.join(base_folder, "images")}/')
    html_doc = html_doc.replace("href='images/", f"href='{os.path.join(base_folder, 'images')}/")

    html_doc = re.sub(r"(?is)<script.*?>.*?</script>", "", html_doc)
    html_doc = re.sub(r'(?is)<link[^>]+rel=["\\\']stylesheet["\\\'][^>]*>', "", html_doc)
    html_doc = re.sub(r"(?is)<style.*?>.*?</style>", "", html_doc)
    html_doc = re.sub(r'(?is)\\sstyle=["\\\'][^"\\\']*["\\\']', "", html_doc)

    title_txt = ""
    m_title = re.search(r'(?is)<div\\s+class=["\\\']title["\\\'][^>]*>(.*?)</div>', html_doc)
    if m_title:
        raw = re.sub(r"(?is)<.*?>", "", m_title.group(1))
        title_txt = raw.strip()
        html_doc = html_doc.replace(m_title.group(0), "")

    html_doc = re.sub(r'(?is)<div\\s+class=["\\\']overlay["\\\'][^>]*>.*?</div>', "", html_doc)
    html_doc = re.sub(r'(?is)<div\\s+class=["\\\']hero["\\\'][^>]*>.*?</div>', "", html_doc)
    html_doc = re.sub(r'(?is)<div\\s+class=["\\\']badges["\\\'].*?>.*?</div>', "", html_doc)

    html_doc = _inline_local_images(html_doc, base_folder)

    css = """
    <style>
      :root { color-scheme: dark; }
      html, body { background: transparent !important; margin:0; padding:0; }
      .nl-wrap, .nl-wrap * { color:#fff !important; }
      .nl-wrap a, .nl-wrap a:visited, .nl-wrap a:hover { color:#fff !important; text-decoration: underline; }
      .nl-wrap li::marker { color:#fff !important; }
      .nl-wrap pre, .nl-wrap code { background: rgba(255,255,255,0.08) !important; color:#fff !important; }
      .nl-wrap img { max-width:100%; height:auto; display:block; border-radius:8px; }
      .nl-wrap * { position: static !important; opacity: 1 !important; filter: none !important; backdrop-filter:none !important; }
      .nl-wrap h1.nl-title { color:#fff !important; margin:.25rem 0 0.75rem;
        font-size: clamp(26px, 3.6vw, 34px) !important; font-weight:800 !important; line-height:1.16 !important; }
      .nl-wrap h2 { font-size: clamp(20px, 3vw, 28px) !important; font-weight:700 !important; }
      .nl-wrap h3 { font-size: clamp(18px, 2.6vw, 24px) !important; font-weight:600 !important; }
      .nl-wrap p  { margin: .45rem 0; line-height: 1.6; font-size: clamp(15px, 2.2vw, 18px); }
    </style>
    """
    title_html = f"<h1 class='nl-title'>{html.escape(title_txt)}</h1>" if title_txt else ""
    body = f"{css}<div class='nl-wrap'>{title_html}{html_doc}</div>"
    iframe_html = f"<!DOCTYPE html><html><head><meta charset='utf-8' /></head><body>{body}</body></html>"
    return iframe_html

def bouton_telecharger_newsletter(nom_fichier, contenu_html):
    st.download_button(
        label="⬇️ Télécharger la Newsletter",
        data=contenu_html,
        file_name=f"{nom_fichier}.html",
        mime="text/html"
    )

# =========================
#    CACHES & CHARGEMENTS
# =========================
@st.cache_data
def charger_urls_et_idees_themes():
    urls = safe_read_csv("urls.csv")
    if urls.empty:
        urls = pd.DataFrame(columns=["titre","date","resume","idees","themes","fichier","url"])

    for col in ("titre","date","resume","idees","themes","fichier","url"):
        if col not in urls.columns: urls[col] = np.nan

    urls = urls.dropna(how="all").copy()

    urls["titre"]   = urls["titre"].fillna("Titre inconnu")
    urls["resume"]  = urls["resume"].fillna("")
    urls["idees"]   = urls["idees"].fillna("")
    urls["themes"]  = urls["themes"].fillna("")
    urls["fichier"] = urls["fichier"].fillna("").astype(str)
    urls["url"]     = urls["url"].fillna("").astype(str)

    def _is_blank(x):
        s = str(x).strip().lower()
        return (s == "") or (s == "nan") or (s == "none") or pd.isna(x)

    essential_cols = ["url", "fichier", "titre", "resume", "themes"]
    urls = urls[~urls[essential_cols].applymap(_is_blank).all(axis=1)].copy()

    def _url_ok(u):
        if _is_blank(u): return False
        u = str(u).strip()
        return ("youtube.com/watch?v=" in u) or ("youtu.be/" in u) or u.startswith("http")
    urls = urls[urls["url"].apply(_url_ok) | ~urls["fichier"].apply(_is_blank)].copy()

    urls["date_str"]  = urls["date"].fillna("Date inconnue").astype(str)
    urls["date_sort"] = pd.to_datetime(urls["date"], errors="coerce")

    idees     = safe_read_csv("idees.csv")
    if "fichier" not in idees.columns: idees["fichier"] = ""
    if "idees"   not in idees.columns: idees["idees"]   = ""
    idees["idees"] = idees["idees"].fillna("")

    idees_v2 = safe_read_csv("idees_v2.csv")

    themes   = _normalize_columns(safe_read_csv("themes.csv"))
    if "fichier" not in themes.columns: themes["fichier"] = ""
    if "themes"  not in themes.columns: themes["themes"]  = ""
    themes["themes"] = themes["themes"].fillna("")

    mesthemes = _normalize_columns(safe_read_csv("mesthemes.csv"))
    mesthemes_list = mesthemes["themes"].dropna().tolist() if "themes" in mesthemes.columns else []

    df = urls.copy()
    if "fichier" in df.columns and "fichier" in idees.columns:
        df = pd.merge(df, idees[["fichier","idees"]], on="fichier", how="left")
    if "fichier" in df.columns and "fichier" in themes.columns:
        df = pd.merge(df, themes[["fichier","themes"]], on="fichier", how="left")

    return df, idees_v2, themes, mesthemes_list, urls

@st.cache_data
def charger_df_seul():
    try:
        df = safe_read_csv("blocs_fusionnes.csv")
        for col in ("url", "start", "text", "fichier"):
            if col not in df.columns:
                df[col] = "" if col != "start" else 0
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des blocs : {e}")
        return pd.DataFrame(columns=["url","start","text","fichier"])

@st.cache_resource
def charger_vecteurs():
    vec = safe_load_pickle("vecteurs.pkl")
    if isinstance(vec, np.ndarray):
        return vec
    if vec is None:
        return np.zeros((0, 1536), dtype="float32")
    try:
        arr = np.array(vec, dtype="float32")
    except Exception:
        arr = np.array(vec)
    return arr

# =========================
#   EMBEDDINGS & SEARCH
# =========================
def embed_openai(query):
    key = get_openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY manquante")
    client = OpenAI(api_key=key)
    return np.array(
        client.embeddings.create(
            input=query, model="text-embedding-3-small", encoding_format="float"
        ).data[0].embedding
    )

def rechercher_similaires(vecteur_query, vecteurs, top_k=5, seuil=0.3):
    if vecteurs is None or len(vecteurs) == 0:
        return np.array([], dtype=int), np.array([])
    try:
        similarities = np.dot(vecteurs, vecteur_query)
    except Exception as e:
        st.error(f"Produit scalaire impossible (dimension ?) : {e}")
        return np.array([], dtype=int), np.array([])
    indices = np.where(similarities >= seuil)[0]
    order = np.argsort(similarities[indices])[::-1][:top_k] if len(indices) else np.array([], dtype=int)
    top_indices = indices[order] if len(indices) else np.array([], dtype=int)
    return top_indices, similarities[top_indices] if len(top_indices) else np.array([])

def build_context(docs):
    """Builds a robust context string from docs, even if metadata is not a dict."""
    parts = []
    for d in (docs or []):
        meta = getattr(d, "metadata", {}) or {}
        url = None
        if isinstance(meta, dict):
            url = meta.get("url") or meta.get("source")
        else:
            # metadata might be a string or something else
            try:
                url = str(meta).strip()
            except Exception:
                url = None
        if not url:
            url = "URL inconnue"
        page = ""
        try:
            page = str(getattr(d, "page_content", "") or "")
        except Exception:
            page = ""
        parts.append(f"[Source: {url}]\n{page}")
    return "\n\n".join(parts)


# =========================
#   DONNÉES LÉGÈRES (toujours chargées)
# =========================
urls_merge_df, idees_v2_df, themes_df, mesthemes_list, urls_df = charger_urls_et_idees_themes()

_all_themes_list = []
if "themes" in themes_df.columns:
    for theme_list in themes_df["themes"].dropna():
        _all_themes_list.extend(_split_and_clean_tags(theme_list))
all_themes = list(dict.fromkeys(_all_themes_list))

url_by_file = {}
if "fichier" in urls_df.columns and "url" in urls_df.columns:
    url_by_file = dict(zip(urls_df["fichier"].astype(str), urls_df["url"].astype(str)))

# =========================
#   NAVIGATION
# =========================
def toggle(key: str):
    st.session_state[key] = not st.session_state.get(key, False)

options = ["🔍 Recherche", "🎥 Toutes les vidéos", "🧠 Moteur intelligent"]
default = st.session_state.get("nav", options[0])
if default not in options: default = options[0]
menu = st.sidebar.radio("Navigation", options, index=options.index(default), key="nav")

# =========================
#   PAGES
# =========================
if menu == "🔍 Recherche":
    df = charger_df_seul()
    vecteurs = charger_vecteurs()

    col1, col2 = st.columns([3, 1])
    if st.session_state.get("reset_search"):
        st.session_state.search_query = ""
        st.session_state.reset_search = False

    with col1:
        st.text_input("🔍 Que veux-tu savoir ?", key="search_query")
    with col2:
        if st.button("🔄 Réinitialiser"):
            st.session_state.selected_theme = ""
            st.session_state.reset_search = True
            do_rerun()

    seuil = st.slider("🌟 Exigence des résultats", 0.1, 0.9, 0.3, 0.05)

    with st.expander("✨ Thèmes", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(mesthemes_list)):
            if cols[i % 4].button(theme, key=f"mestheme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.reset_search = True
                do_rerun()

    with st.expander("🏷️ Tags", expanded=False):
        render_tags_scroller_interactive(sorted(all_themes), uid="global-tags", height=136)

    query = to_str(st.session_state.get("search_query", "")).strip() or to_str(st.session_state.get("selected_theme", "")).strip()

    if query:
        try:
            with st.spinner("🔍 Recherche en cours..."):
                try:
                    vecteur_query = embed_openai(query)
                except Exception as e_embed:
                    st.error(f"Échec embedding OpenAI : {e_embed}")
                    vecteur_query = None

                if vecteur_query is not None:
                    if vecteurs is not None and vecteurs.size > 0:
                        if vecteurs.shape[1] != len(vecteur_query):
                            st.error(f"Dimension vecteurs ({vecteurs.shape[1]}) ≠ embedding ({len(vecteur_query)})")
                            indices, scores = np.array([], dtype=int), np.array([])
                        else:
                            indices, scores = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)
                    else:
                        indices, scores = np.array([], dtype=int), np.array([])
                else:
                    indices, scores = np.array([], dtype=int), np.array([])
        except Exception as e:
            st.error(f"Erreur recherche : {e}")
            indices, scores = np.array([], dtype=int), np.array([])

        if len(indices) == 0:
            st.warning("Aucun résultat trouvé ou moteur IA indisponible.")
        else:
            st.markdown("### 🌟 Résultats pertinents :")
            for idx, score in zip(indices, scores):
                bloc = df.iloc[int(idx)]
                url_str = to_str(bloc.get("url", ""))
                if not url_str:
                    fichier_key = to_str(bloc.get("fichier", ""))
                    url_str = url_by_file.get(fichier_key, "")

                youtube_id = extract_youtube_id(url_str)
                start_time = to_int(bloc.get("start", 0), 0)
                text = to_str(bloc.get("text", ""))

                with st.expander(f"⏱️ {start_time}s — 💬 {text[:60]}... (score: {score:.2f})"):
                    if text: st.markdown(f"**Texte complet :** {text}")
                    if youtube_id:
                        embed_url = f"https://www.youtube.com/embed/{youtube_id}?start={start_time}&autoplay=0"
                        st.components.v1.iframe(embed_url, height=315)
                    elif url_str:
                        st.markdown(f"[▶️ Ouvrir la vidéo]({url_str})")
                    else:
                        st.info("Aucune URL vidéo disponible pour ce résultat.")

elif menu == "🎥 Toutes les vidéos":
    st.header("📚 Liste des vidéos disponibles")

    cols_refresh = st.columns([1, 3])
    with cols_refresh[0]:
        if st.button("🔄 Actualiser les vidéos"):
            st.cache_data.clear()
            do_rerun()

    st.text_input("🔍 Recherche par titre, résumé, idée ou thème", key="video_search")
    tri = st.selectbox("📜 Trier par", ("Date récente", "Date ancienne", "Titre A → Z", "Titre Z → A"))

    recherche = to_str(st.session_state.get("video_search", "")).strip()
    urls_view = urls_df.copy()

    if recherche:
        urls_view = urls_view[urls_view.apply(
            lambda row: recherche.lower() in (to_str(row.get("titre","")) +
                                      to_str(row.get("resume","")) +
                                      to_str(row.get("idees","")) +
                                      to_str(row.get("themes","")) +
                                      to_str(row.get("fichier",""))).lower(),
            axis=1
        )]

    if "date_sort" not in urls_view.columns:
        urls_view["date_sort"] = pd.to_datetime(urls_view.get("date", None), errors="coerce")

    if tri == "Date récente":
        urls_view = urls_view.sort_values(["date_sort","titre"], ascending=[False, True], na_position="last")
    elif tri == "Date ancienne":
        urls_view = urls_view.sort_values(["date_sort","titre"], ascending=[True, True], na_position="last")
    elif tri == "Titre A → Z":
        urls_view = urls_view.sort_values("titre", ascending=True, na_position="last")
    elif tri == "Titre Z → A":
        urls_view = urls_view.sort_values("titre", ascending=False, na_position="last")

    st.markdown(f"### 🎬 {len(urls_view)} vidéo(s) trouvée(s)")

    for _, row in urls_view.iterrows():
        video_name  = to_str(row.get("titre", "Titre inconnu"))
        video_date  = to_str(row.get("date_str", row.get("date", "Date inconnue")))
        fichier_nom = to_str(row.get("fichier", ""))
        primary_title = fichier_nom if fichier_nom else video_name

        url_str = to_str(row.get("url", ""))
        if not url_str and fichier_nom:
            url_str = url_by_file.get(fichier_nom, "")

        resume = to_str(row.get("resume", ""))
        idees  = to_str(row.get("idees", ""))
        themes = to_str(row.get("themes", ""))

        youtube_id = extract_youtube_id(url_str)

        if (to_str(url_str) == "") and (video_name == "Titre inconnu"):
            continue

        col1, col2 = st.columns([1, 5])
        with col1:
            if youtube_id:
                thumbnail_url = f"https://img.youtube.com/vi/{youtube_id}/0.jpg"
                show_image(thumbnail_url, width=140)
            else:
                st.write("🖼️ Miniature indisponible")

        with col2:
            if url_str: st.markdown(f"### [{primary_title}]({url_str})")
            else:       st.markdown(f"### {primary_title}")
            meta_line = f"🗓️ <em>{video_date}</em> — <span style='font-size:0.95rem; opacity:0.85'>{video_name}</span>"
            st.markdown(meta_line, unsafe_allow_html=True)

            if resume: st.markdown(f"📜 {resume}")

            if fichier_nom:
                state_key = f"show_newsletter_{fichier_nom}"
                if st.button("📬 Newsletter liée à cette vidéo", key=f"btn_nl_{fichier_nom}"):
                    toggle(state_key)

                if st.session_state.get(state_key, False):
                    newsletter_contenu = charger_newsletter_html(fichier_nom)
                    if newsletter_contenu:
                        newsletter_contenu = fix_newsletter_html(newsletter_contenu, NEWSLETTER_DIR)
                        with st.expander("📬 Newsletter (ouvrir/fermer)", expanded=True):
                            st.components.v1.html(newsletter_contenu, height=900, scrolling=True)
                            bouton_telecharger_newsletter(fichier_nom, newsletter_contenu)
                    else:
                        st.warning("❌ Pas de newsletter disponible pour cette vidéo.")

            if themes:
                render_tags_scroller(themes, uid=(fichier_nom or 'tags'))

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            if idees:
                with st.expander("🌟 Sujets de la vidéo"):
                    for idee in idees.split("|"):
                        i = idee.strip()
                        if i and youtube_id:
                            st.markdown(f"- [{i}](https://www.youtube.com/watch?v={youtube_id}&t=0s)")
                        elif i:
                            st.markdown(f"- {i}")

            if fichier_nom and "fichier" in idees_v2_df.columns:
                with st.expander("🕒 Moments de la vidéo"):
                    idees_v2_video = idees_v2_df[idees_v2_df["fichier"] == fichier_nom]
                    for _, idee_row in idees_v2_video.iterrows():
                        idee_text = to_str(idee_row.get("idee", ""))
                        start_time = to_int(idee_row.get("start", 0), 0)
                        if idee_text and youtube_id:
                            st.markdown(f"- [{idee_text}](https://www.youtube.com/watch?v={youtube_id}&t={start_time}s)")
                        elif idee_text:
                            st.markdown(f"- {idee_text}")

        st.markdown("---")

elif menu == "🧠 Moteur intelligent":
    st.header("🧠 Assistant IA basé sur vos formations vidéos")
    st.text_input("Pose ta question", key="user_question")
    user_question = to_str(st.session_state.get("user_question", "")).strip()

    if DEBUG_MODE:
        st.info("DEBUG: page IA chargée")

    if user_question:
        df = charger_df_seul()
        vecteurs = charger_vecteurs()
        try:
            if DEBUG_MODE: st.write("DEBUG: question reçue →", user_question)

            with st.spinner("Recherche intelligente en cours..."):
                docs = None

                if not FORCE_NUMPY:
                    try:
                        if DEBUG_MODE: st.write("DEBUG: tentative FAISS.load_local(...)")
                        vectordb = FAISS.load_local(
                            FAISS_DIR,
                            OpenAIEmbeddings(openai_api_key=get_openai_key() or ""),
                            allow_dangerous_deserialization=True
                        )
                        if DEBUG_MODE: st.write("DEBUG: FAISS chargé, similarity_search()…")
                        docs = vectordb.similarity_search(user_question, k=5)
                        if DEBUG_MODE: st.write("DEBUG: résultats FAISS →", len(docs))
                    except Exception as e_faiss:
                        st.warning(f"⚠️ FAISS indisponible: {e_faiss}\n→ bascule sur numpy")

                if docs is None:
                    if DEBUG_MODE: st.write("DEBUG: Fallback numpy → embed_openai()")
                    try:
                        vq = embed_openai(user_question)
                    except Exception as e_embed:
                        st.error(f"Échec embedding OpenAI : {e_embed}")
                        vq = None

                    if vq is None or vecteurs is None or vecteurs.size == 0:
                        st.error("Aucun vecteur ou embedding indisponible.")
                        docs = []
                    else:
                        if vecteurs.shape[1] != len(vq):
                            st.error(f"Dimension vecteurs ({vecteurs.shape[1]}) ≠ embedding ({len(vq)})")
                            docs = []
                        else:
                            idxs, _ = rechercher_similaires(vq, vecteurs, top_k=5, seuil=0.25)
                            if DEBUG_MODE: st.write("DEBUG: indices trouvés →", list(map(int, idxs)))

                            class Doc:
                                def __init__(self, page_content, metadata):
                                    self.page_content = page_content
                                    self.metadata = metadata

                            docs = []
                            for i in idxs:
                                r = df.iloc[int(i)]
                                url_str = to_str(r.get("url", "")) or url_by_file.get(to_str(r.get("fichier","")), "")
                                docs.append(Doc(page_content=to_str(r.get("text","")), metadata={"url": url_str}))
                            if DEBUG_MODE: st.write("DEBUG: docs construits →", len(docs))

                context = build_context(docs)
                if DEBUG_MODE: st.write("DEBUG: longueur contexte →", len(context))

                if not context.strip():
                    st.error("Je n'ai pas trouvé cette information dans notre base actuelle.")
                else:
                    if DEBUG_MODE: st.write("DEBUG: appel ChatOpenAI.invoke()")
                    key = get_openai_key()
                    if not key:
                        st.error("Clé OpenAI manquante pour la réponse IA.")
                    else:
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=key)
                        prompt = f"""Tu es un expert de notre entreprise. Voici des extraits de nos formations :

{context}

Réponds précisément à la question suivante en utilisant uniquement ces extraits.
Si aucune information n'existe, réponds : "Je n'ai pas trouvé cette information dans notre base actuelle."

Question : {user_question}
""".strip()
                        resp = llm.invoke(prompt)
                        if DEBUG_MODE: st.write("DEBUG: réponse reçue")
                        st.success(resp.content)

        except Exception as e:
            st.error("Une erreur est survenue dans la recherche intelligente.")
            st.exception(e)
    else:
        st.info("Saisis une question pour lancer la recherche.")
