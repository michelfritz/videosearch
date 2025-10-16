from openai import OpenAI
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
import chardet
import re, html, urllib.parse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# -----------------------------------------------------------------------------
# OpenAI key (inchangé)
# -----------------------------------------------------------------------------
def get_openai_key():
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Clé OpenAI absente. Définis OPENAI_API_KEY (Cloud Run: Variables & secrets).")
        raise RuntimeError("OPENAI_API_KEY manquant")
    return key

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")
st.caption(f"Clé OpenAI détectée : {bool(get_openai_key())}")

# -----------------------------------------------------------------------------
# Helpers CSV & tags (inchangé)
# -----------------------------------------------------------------------------
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
    if value is None:
        return []
    s = str(value)
    for sep in [" ; ", ";", ",", " / ", "/", "\\", "  "]:
        s = s.replace(sep, "|")
    parts = [p.strip() for p in s.split("|")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

# -----------------------------------------------------------------------------
# Session, utils (inchangé)
# -----------------------------------------------------------------------------
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
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

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
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)

def extract_youtube_id(url) -> str:
    s = to_str(url, "")
    if not s:
        return ""
    s = s.strip()
    if "watch?v=" in s:
        part = s.split("watch?v=")[-1]
        return part.split("&")[0]
    if "youtu.be/" in s:
        part = s.split("youtu.be/")[-1]
        return part.split("?")[0]
    return ""

def to_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(float(x))
    except Exception:
        return default

# -----------------------------------------------------------------------------
# Tags visuels pour les cartes vidéo (inchangé)
# -----------------------------------------------------------------------------
def render_tags_scroller(themes_str: str, uid: str, height: int = 120):
    if not themes_str:
        return
    chips = [t.strip() for t in str(themes_str).split("|") if t.strip()]
    if not chips:
        return
    items_html = "".join([f"<span class='tag'>{html.escape(c)}</span>" for c in chips])
    html_block = """
    <style>
      .tagbox-{uid} .bar{{display:flex;align-items:center;gap:.25rem;margin:.25rem 0;}}
      .tagbox-{uid} .wrap{{display:grid;grid-auto-flow:column;grid-template-rows:repeat(2,auto);
                           gap:8px 8px;overflow-x:auto;overflow-y:hidden;padding:6px 6px;
                           scroll-behavior:smooth; -webkit-overflow-scrolling:touch; scrollbar-width:thin;}}
      .tagbox-{uid} .wrap::-webkit-scrollbar{{height:8px}}
      .tagbox-{uid} .wrap::-webkit-scrollbar-thumb{{background:rgba(0,0,0,.15);border-radius:8px}}
      .tagbox-{uid} .tag{{display:inline-flex;align-items:center;justify-content:center;text-align:center;
                          white-space:nowrap;background:#D0E8FF;color:#0A2540;
                          border-radius:999px;padding:6px 14px;font-size:13px;border:1px solid rgba(0,0,0,.05);}}
      .tagbox-{uid} .btn{{border:0;background:transparent;color:#6b7280;font-size:22px;cursor:pointer;
                          padding:0 6px;line-height:1}}
    </style>
    <div class="tagbox-{uid}">
      <div class="bar">
        <button class="btn" onclick="document.getElementById('wrap-{uid}').scrollBy({{left:-320,behavior:'smooth'}})">&#9664;</button>
        <div id="wrap-{uid}" class="wrap" style="height:{h}px">{items_html}</div>
        <button class="btn" onclick="document.getElementById('wrap-{uid}').scrollBy({{left:320,behavior:'smooth'}})">&#9654;</button>
      </div>
    </div>
    """.format(uid=uid, h=height-24, items_html=items_html)
    st.components.v1.html(html_block, height=height, scrolling=False)

# -----------------------------------------------------------------------------
# *** NOUVEAU ***  Tag picker SANS rechargement de page
#   - composant HTML (iframe) qui renvoie le tag cliqué via Streamlit.setComponentValue
#   - met à jour selected_theme et relance le script
# -----------------------------------------------------------------------------
def render_tag_picker(tags: list[str], uid: str = "global-tags", height: int = 136):
    if not tags:
        return
    chips = "".join(
        f"<button type='button' class='tag' data-tag='{html.escape(t)}'>{html.escape(t)}</button>"
        for t in tags
    )
    html_code = f"""
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
         background:#D0E8FF;color:#0A2540;border-radius:999px;padding:6px 14px;font-size:13px;border:1px solid rgba(0,0,0,.05);
         cursor:pointer; user-select:none;
      }}
      .sbox-{uid} .btn{{border:0;background:transparent;color:#6b7280;font-size:22px;cursor:pointer;padding:0 6px;line-height:1}}
    </style>
    <div class="sbox-{uid}">
      <div class="bar">
        <button class="btn" id="prev-{uid}">&#9664;</button>
        <div id="wrap-{uid}" class="wrap" style="height:{height-24}px">{chips}</div>
        <button class="btn" id="next-{uid}">&#9654;</button>
      </div>
    </div>
    <script>
      const wrap = document.getElementById("wrap-{uid}");
      const prev = document.getElementById("prev-{uid}");
      const next = document.getElementById("next-{uid}");
      if (prev && next && wrap) {{
        prev.onclick = () => wrap.scrollBy({{left:-320, behavior:"smooth"}});
        next.onclick = () => wrap.scrollBy({{left:320,  behavior:"smooth"}});
      }}
      (wrap || document).querySelectorAll(".tag").forEach(el => {{
        el.addEventListener("click", (e) => {{
          e.preventDefault(); e.stopPropagation();
          const v = el.getAttribute("data-tag") || el.textContent.trim();
          if (window.Streamlit && v) window.Streamlit.setComponentValue(v);
        }});
      }});
    </script>
    """
    picked = st.components.v1.html(html_code, height=height, scrolling=False)
    if picked:
        st.session_state.selected_theme = str(picked)
        st.session_state.nav = "🔍 Recherche"
        st.session_state.reset_search = True
        do_rerun()

# -----------------------------------------------------------------------------
# Support éventuel du paramètre ?select_tag=... (laisse inchangé le deep‑link)
# -----------------------------------------------------------------------------
def handle_select_tag_from_query():
    qp = st.query_params
    tag = qp.get("select_tag", None)
    if isinstance(tag, list):
        tag = tag[0] if tag else None
    if tag:
        st.session_state.selected_theme = tag
        st.session_state.nav = "🔍 Recherche"
        try:
            qp.pop("select_tag", None)
        except Exception:
            pass
        st.session_state.reset_search = True

# -----------------------------------------------------------------------------
# Données & newsletters (inchangé)
# -----------------------------------------------------------------------------
show_image("logo_lucarne.png", width=180)
st.markdown("# 📚 Base de connaissance A LA LUCARNE")
openai.api_key = os.environ.get("OPENAI_API_KEY")

DOSSIER_NEWSLETTERS = "newsletters"

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def charger_newsletter_html(nom_fichier):
    chemin = os.path.join(DOSSIER_NEWSLETTERS, f"{nom_fichier}.html")
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return f.read()
    return None

def bouton_telecharger_newsletter(nom_fichier, contenu_html):
    st.download_button(
        label="⬇️ Télécharger la Newsletter",
        data=contenu_html,
        file_name=f"{nom_fichier}.html",
        mime="text/html"
    )

@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    for col in ("url", "start", "text", "fichier"):
        if col not in df.columns:
            df[col] = "" if col != "start" else 0
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls_et_idees_themes():
    urls = pd.read_csv("urls.csv", encoding=detect_encoding("urls.csv"))
    urls = urls.replace(r"^\s*(nan|null|none|NaN)\s*$", np.nan, regex=True)

    for col in ("titre","date","resume","idees","themes","fichier","url"):
        if col not in urls.columns:
            urls[col] = np.nan
    urls = urls.dropna(how="all")

    def _is_blank(x):
        s = str(x).strip().lower()
        return (s == "") or (s == "nan") or (s == "none") or pd.isna(x)

    essential_cols = ["url", "fichier", "titre", "resume", "themes"]
    urls = urls[~urls[essential_cols].applymap(_is_blank).all(axis=1)].copy()
    urls = urls[~(urls["url"].apply(_is_blank) & urls["fichier"].apply(_is_blank))].copy()

    def _url_ok(u):
        if _is_blank(u): return False
        u = str(u).strip()
        return ("youtube.com/watch?v=" in u) or ("youtu.be/" in u) or u.startswith("http")
    urls = urls[urls["url"].apply(_url_ok) | ~urls["fichier"].apply(_is_blank)].copy()

    urls["titre"]  = urls["titre"].fillna("Titre inconnu")
    urls["date"]   = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")
    urls["idees"]  = urls["idees"].fillna("")
    urls["themes"] = urls["themes"].fillna("")
    urls["fichier"]= urls["fichier"].fillna("").astype(str)
    urls["url"]    = urls["url"].fillna("").astype(str)

    idees = pd.read_csv("idees.csv", encoding=detect_encoding("idees.csv"))
    if "fichier" not in idees.columns: idees["fichier"] = ""
    if "idees" not in idees.columns:   idees["idees"] = ""
    idees["idees"] = idees["idees"].fillna("")

    idees_v2 = pd.read_csv("idees_v2.csv", encoding=detect_encoding("idees_v2.csv"))

    themes = _normalize_columns(pd.read_csv("themes.csv", encoding=detect_encoding("themes.csv")))
    if "fichier" not in themes.columns: themes["fichier"] = ""
    if "themes" not in themes.columns:  themes["themes"] = ""
    themes["themes"] = themes["themes"].fillna("")

    mesthemes = _normalize_columns(pd.read_csv("mesthemes.csv", encoding=detect_encoding("mesthemes.csv")))
    mesthemes_list = mesthemes["themes"].dropna().tolist() if "themes" in mesthemes.columns else []

    df = urls.copy()
    if "fichier" in df.columns and "fichier" in idees.columns:
        df = pd.merge(df, idees[["fichier","idees"]], on="fichier", how="left")
    if "fichier" in df.columns and "fichier" in themes.columns:
        df = pd.merge(df, themes[["fichier","themes"]], on="fichier", how="left")

    return df, idees_v2, themes, mesthemes_list

# Embeddings + recherche (inchangé)
def embed_openai(query):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small",
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

def rechercher_similaires(vecteur_query, vecteurs, top_k=5, seuil=0.3):
    similarities = np.dot(vecteurs, vecteur_query)
    indices = np.where(similarities >= seuil)[0]
    top_indices = indices[np.argsort(similarities[indices])[::-1][:top_k]]
    return top_indices, similarities[top_indices]

# =====================
#       DONNÉES
# =====================
init_state()
handle_select_tag_from_query()

df, vecteurs = charger_donnees()
urls_df, idees_v2_df, themes_df, mesthemes_list = charger_urls_et_idees_themes()

_all_themes_list = []
if "themes" in themes_df.columns:
    for theme_list in themes_df["themes"].dropna():
        _all_themes_list.extend(_split_and_clean_tags(theme_list))
all_themes = list(dict.fromkeys(_all_themes_list))

url_by_file = {}
if "fichier" in urls_df.columns and "url" in urls_df.columns:
    url_by_file = dict(zip(urls_df["fichier"].astype(str), urls_df["url"].astype(str)))

# -----------------------------------------------------------------------------
# Newsletter – nettoyage & style (inchangé)
# -----------------------------------------------------------------------------
def fix_newsletter_html(html_src: str, base_folder=DOSSIER_NEWSLETTERS) -> str:
    html_doc = html_src or ""
    if not html_doc:
        return ""
    html_doc = html_doc.replace('src="images/', f'src="{base_folder}/images/')
    html_doc = html_doc.replace("src='images/", f"src='{base_folder}/images/")
    html_doc = html_doc.replace('href="images/', f'href="{base_folder}/images/')
    html_doc = html_doc.replace("href='images/", f"href='{base_folder}/images/")

    html_doc = re.sub(r"(?is)<script.*?>.*?</script>", "", html_doc)
    html_doc = re.sub(r'(?is)<link[^>]+rel=["\']stylesheet["\'][^>]*>', "", html_doc)
    html_doc = re.sub(r"(?is)<style.*?>.*?</style>", "", html_doc)
    html_doc = re.sub(r'(?is)\sstyle=["\'][^"\']*["\']', "", html_doc)

    title_txt = ""
    m_title = re.search(r'(?is)<div\s+class=["\']title["\'][^>]*>(.*?)</div>', html_doc)
    if m_title:
        raw = re.sub(r"(?is)<.*?>", "", m_title.group(1))
        title_txt = raw.strip()
        html_doc = html_doc.replace(m_title.group(0), "")

    html_doc = re.sub(r'(?is)<div\s+class=["\']overlay["\'][^>]*>.*?</div>', "", html_doc)
    html_doc = re.sub(r'(?is)<div\s+class=["\']hero["\'][^>]*>.*?</div>', "", html_doc)
    html_doc = re.sub(r'(?is)<div\s+class=["\']badges["\'].*?>.*?</div>', "", html_doc)

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
      .nl-wrap h1.nl-title { 
        margin:.25rem 0 .75rem;
        font-size: clamp(24px, 3.2vw, 30px) !important; font-weight:800 !important; line-height:1.16 !important;
      }
      .nl-wrap h2 { font-size: clamp(20px, 3vw, 28px) !important; font-weight:700 !important; }
      .nl-wrap h3 { font-size: clamp(18px, 2.6vw, 24px) !important; font-weight:600 !important; }
      .nl-wrap p  { margin: .45rem 0; line-height: 1.6; font-size: clamp(15px, 2.2vw, 18px); }
    </style>
    """
    title_html = f"<h1 class='nl-title'>{title_txt}</h1>" if title_txt else ""
    body = f"{css}<div class='nl-wrap'>{title_html}{html_doc}</div>"
    return f"<!DOCTYPE html><html><head><meta charset='utf-8' /></head><body>{body}</body></html>"

def toggle(key: str):
    st.session_state[key] = not st.session_state.get(key, False)

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
options = ["🔍 Recherche", "🎥 Toutes les vidéos", "🧠 Moteur intelligent"]
default = st.session_state.get("nav", options[0])
if default not in options:
    default = options[0]
menu = st.sidebar.radio("Navigation", options, index=options.index(default), key="nav")

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
if menu == "🔍 Recherche":
    col1, col2 = st.columns([3, 1])

    if st.session_state.reset_search:
        st.session_state.search_query = ""
        st.session_state.reset_search = False

    with col1:
        st.text_input("🔍 Que veux-tu savoir ?", key="search_query")
    with col2:
        if st.button("🔄 Réinitialiser"):
            st.session_state.selected_theme = ""
            st.session_state.reset_search = True
            do_rerun()

    seuil = st.slider("🌟 Exigence des résultats", 0.1, 0.9, 0.5, 0.05)

    # Thèmes perso – inchangé
    with st.expander("✨ Thèmes", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(mesthemes_list)):
            if cols[i % 4].button(theme, key=f"mestheme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.reset_search = True
                do_rerun()

    # 👉 Tags globaux : même style visuel, mais clic SANS rechargement
    with st.expander("🏷️ Tags", expanded=False):
        render_tag_picker(sorted(all_themes), uid="global-tags", height=136)

    # Construire la requête
    query = to_str(st.session_state.get("search_query", "")).strip() or \
            to_str(st.session_state.get("selected_theme", "")).strip()

    if query:
        with st.spinner("🔍 Recherche en cours..."):
            vecteur_query = embed_openai(query)
            indices, scores = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)

        if len(indices) == 0:
            st.warning("Aucun résultat trouvé.")
        else:
            st.markdown("### 🌟 Résultats pertinents :")
            for idx, score in zip(indices, scores):
                bloc = df.iloc[idx]
                url_str = to_str(bloc.get("url", ""))
                if not url_str:
                    fichier_key = to_str(bloc.get("fichier", ""))
                    url_str = url_by_file.get(fichier_key, "")

                youtube_id = extract_youtube_id(url_str)
                start_time = to_int(bloc.get("start", 0), 0)
                text = to_str(bloc.get("text", ""))

                with st.expander(f"⏱️ {start_time}s — 💬 {text[:60]}... (score: {score:.2f})"):
                    if text:
                        st.markdown(f"**Texte complet :** {text}")
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
                                              to_str(row.get("themes",""))).lower(),
            axis=1
        )]

    if "date" in urls_view.columns:
        if tri == "Date récente":
            urls_view = urls_view.sort_values("date", ascending=False)
        elif tri == "Date ancienne":
            urls_view = urls_view.sort_values("date", ascending=True)

    if "titre" in urls_view.columns:
        if tri == "Titre A → Z":
            urls_view = urls_view.sort_values("titre", ascending=True)
        elif tri == "Titre Z → A":
            urls_view = urls_view.sort_values("titre", ascending=False)

    st.markdown(f"### 🎬 {len(urls_view)} vidéo(s) trouvée(s)")

    for _, row in urls_view.iterrows():
        video_name  = to_str(row.get("titre", "Titre inconnu"))
        video_date  = to_str(row.get("date", "Date inconnue"))
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
            if url_str:
                st.markdown(f"### [{primary_title}]({url_str})")
            else:
                st.markdown(f"### {primary_title}")
            meta_line = f"🗓️ <em>{video_date}</em> — <span style=\"font-size:0.95rem; opacity:0.85\">{video_name}</span>"
            st.markdown(meta_line, unsafe_allow_html=True)

            if resume:
                st.markdown(f"📜 {resume}")

            if fichier_nom:
                state_key = f"show_newsletter_{fichier_nom}"
                if st.button("📬 Newsletter liée à cette vidéo", key=f"btn_nl_{fichier_nom}"):
                    toggle(state_key)
                if st.session_state.get(state_key, False):
                    newsletter_contenu = charger_newsletter_html(fichier_nom)
                    if newsletter_contenu:
                        newsletter_contenu = fix_newsletter_html(newsletter_contenu)
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

    st.text_input("Pose ta question :", key="user_question")
    user_question = to_str(st.session_state.get("user_question", "")).strip()

    if user_question:
        with st.spinner("Recherche intelligente en cours..."):
            vectordb = FAISS.load_local(
                "faiss_transcripts",
                OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")),
                allow_dangerous_deserialization=True
            )
            docs = vectordb.similarity_search(user_question, k=5)
            context = ""
            for doc in docs:
                url = doc.metadata.get("url", "URL inconnue")
                context += f"[Source: {url}]\n{doc.page_content}\n\n"

            prompt = f"""
Tu es un expert de notre entreprise. Voici des extraits de nos formations :

{context}

Réponds précisément à la question suivante en utilisant uniquement ces extraits.
Si aucune information n'existe, réponds : "Je n'ai pas trouvé cette information dans notre base actuelle."

Question : {user_question}
"""
            llm = ChatOpenAI(
                model="gpt-4-0125-preview",
                temperature=0.2,
                openai_api_key=openai.api_key
            )
            response = llm.invoke(prompt)
            st.success(response.content)
