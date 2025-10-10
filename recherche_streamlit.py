from openai import OpenAI
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
import chardet
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# ------------------------------------
# Page setup
# ------------------------------------
st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")

# ------------------------------------
# Helpers: session state + safe image + safe URL + safe rerun
# ------------------------------------
def init_state():
    """Initialize all session_state keys that are later read by the app."""
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
    # Works on both new/old Streamlit versions
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def show_image(img, width=None, caption=None):
    """Robust wrapper for st.image that avoids width=0 and empty/invalid sources."""
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

init_state()

# 🎨 Logo (optionnel)
show_image("logo_lucarne.png", width=180)
st.markdown("# 📚 Base de connaissance A LA LUCARNE")

# 🔐 Clé API OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ------------------------------------
# Data loading + encoding helpers
# ------------------------------------
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
    # Sécuriser colonnes utilisées par la page Recherche
    for col in ("url", "start", "text", "fichier"):
        if col not in df.columns:
            df[col] = "" if col != "start" else 0
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls_et_idees_themes():
    # — lecture robuste + normalisation des 'nan' en NaN réels
    urls = pd.read_csv("urls.csv", encoding=detect_encoding("urls.csv"))
    urls = urls.replace(r"^\s*(nan|null|none|NaN)\s*$", np.nan, regex=True)

    # colonnes attendues
    for col in ("titre","date","resume","idees","themes","fichier","url"):
        if col not in urls.columns:
            urls[col] = np.nan

    # drop lignes totalement vides
    urls = urls.dropna(how="all")

    # helper pour tester le “vide” (NaN, '', 'nan', espaces)
    def _is_blank(x):
        s = str(x).strip().lower()
        return (s == "") or (s == "nan") or (s == "none") or pd.isna(x)

    # on ne garde que les lignes avec AU MOINS un champ utile
    essential_cols = ["url", "fichier", "titre", "resume", "themes"]
    urls = urls[~urls[essential_cols].applymap(_is_blank).all(axis=1)].copy()

    # Nettoyage basique : si 'url' est vide ET 'fichier' est vide → ignorer
    urls = urls[~(urls["url"].apply(_is_blank) & urls["fichier"].apply(_is_blank))].copy()

    # Optionnel : ne garder que des URLs plausibles (YouTube ou http(s))
    def _url_ok(u):
        if _is_blank(u): return False
        u = str(u).strip()
        return ("youtube.com/watch?v=" in u) or ("youtu.be/" in u) or u.startswith("http")
    urls = urls[urls["url"].apply(_url_ok) | ~urls["fichier"].apply(_is_blank)].copy()

    # Valeurs par défaut (affichage)
    urls["titre"]  = urls["titre"].fillna("Titre inconnu")
    urls["date"]   = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")
    urls["idees"]  = urls["idees"].fillna("")
    urls["themes"] = urls["themes"].fillna("")
    urls["fichier"]= urls["fichier"].fillna("").astype(str)
    urls["url"]    = urls["url"].fillna("").astype(str)

    # --- idem qu’avant pour les autres CSV ---
    idees = pd.read_csv("idees.csv", encoding=detect_encoding("idees.csv"))
    if "fichier" not in idees.columns: idees["fichier"] = ""
    if "idees" not in idees.columns:   idees["idees"] = ""
    idees["idees"] = idees["idees"].fillna("")

    idees_v2 = pd.read_csv("idees_v2.csv", encoding=detect_encoding("idees_v2.csv"))

    themes = pd.read_csv("themes.csv", encoding=detect_encoding("themes.csv"))
    if "fichier" not in themes.columns: themes["fichier"] = ""
    if "themes" not in themes.columns:  themes["themes"] = ""
    themes["themes"] = themes["themes"].fillna("")

    mesthemes = pd.read_csv("mesthemes.csv", encoding=detect_encoding("mesthemes.csv"))
    mesthemes_list = mesthemes["themes"].dropna().tolist() if "themes" in mesthemes.columns else []

    # merges robustes
    df = urls.copy()
    if "fichier" in df.columns and "fichier" in idees.columns:
        df = pd.merge(df, idees[["fichier","idees"]], on="fichier", how="left")
    if "fichier" in df.columns and "fichier" in themes.columns:
        df = pd.merge(df, themes[["fichier","themes"]], on="fichier", how="left")

    return df, idees_v2, themes, mesthemes_list

# ------------------------------------
# Embeddings + vector search
# ------------------------------------
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
df, vecteurs = charger_donnees()
urls_df, idees_v2_df, themes_df, mesthemes_list = charger_urls_et_idees_themes()

# 🔖 Préparer tous les thèmes
all_themes = set()
if "themes" in themes_df.columns:
    for theme_list in themes_df["themes"].dropna():
        for theme in to_str(theme_list).split("|"):
            theme = theme.strip()
            if theme:
                all_themes.add(theme)

# 🔁 Fallback URL par 'fichier' (utile si df["url"] est vide)
url_by_file = {}
if "fichier" in urls_df.columns and "url" in urls_df.columns:
    url_by_file = dict(zip(urls_df["fichier"].astype(str), urls_df["url"].astype(str)))

# ---- Sidebar Navigation (robuste) ----
options = ["🔍 Recherche", "🎥 Toutes les vidéos", "🧠 Moteur intelligent"]
default = st.session_state.get("nav", options[0])
if default not in options:
    default = options[0]
menu = st.sidebar.radio("Navigation", options, index=options.index(default), key="nav")

def fix_newsletter_html(html: str, base_folder=DOSSIER_NEWSLETTERS) -> str:
    """
    - Réécrit les chemins relatifs vers le sous-dossier 'newsletters/images/...'
    - Injecte un peu de CSS pour les badges si la feuille externe n'est pas dispo.
    """
    if not html:
        return html

    # Normaliser les chemins d'images (src="images/...") -> src="newsletters/images/..."
    html = html.replace('src="images/', f'src="{base_folder}/images/')
    html = html.replace("src='images/", f"src='{base_folder}/images/")

    # Idem pour href éventuels (liens vers images ou assets)
    html = html.replace('href="images/', f'href="{base_folder}/images/')
    html = html.replace("href='images/", f"href='{base_folder}/images/")

    # CSS minimal pour les badges si .badges est utilisé
    css = """
    <style>
      .badges{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0;}
      .badges span{background:#EEF6FF;border:1px solid #CDE3FF;border-radius:16px;
                   padding:.25rem .6rem;font-size:.85rem;}
      .nl-hero img{max-width:100%;height:auto;border-radius:8px;display:block;}
    </style>
    """
    # Injecter la CSS au début si absent
    if "<style" not in html[:800]:
        html = css + html
    return html


def toggle(key: str):
    st.session_state[key] = not st.session_state.get(key, False)


# =====================
#       PAGES
# =====================
if menu == "🔍 Recherche":
    col1, col2 = st.columns([3, 1])

    # Réinitialiser si besoin
    if st.session_state.reset_search:
        st.session_state.search_query = ""
        st.session_state.reset_search = False

    # Champ de recherche
    with col1:
        st.text_input("🔍 Que veux-tu savoir ?", key="search_query")

    # Bouton Réinitialiser
    with col2:
        if st.button("🔄 Réinitialiser"):
            st.session_state.selected_theme = ""
            st.session_state.reset_search = True
            do_rerun()

    seuil = st.slider("🌟 Exigence des résultats", 0.1, 0.9, 0.5, 0.05)

    # 🌟 Mes Thèmes personnalisés
    with st.expander("✨ Thèmes", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(mesthemes_list)):
            if cols[i % 4].button(theme, key=f"mestheme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.reset_search = True
                do_rerun()

    # 🌟 Tous les Thèmes
    with st.expander("🏷️ Tags", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(all_themes)):
            if cols[i % 4].button(theme, key=f"theme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.reset_search = True
                do_rerun()

    # Définir la requête
    query = to_str(st.session_state.get("search_query", "")).strip() or to_str(st.session_state.get("selected_theme", "")).strip()

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

                # URL prioritaire depuis le bloc; sinon fallback par 'fichier'
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

    # 🔄 bouton de refresh data (invalide le cache, recharge les CSV)
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
        video_name = to_str(row.get("titre", "Titre inconnu"))
        video_date = to_str(row.get("date", "Date inconnue"))
        fichier_nom = to_str(row.get("fichier", ""))

        # URL + fallback
        url_str = to_str(row.get("url", ""))
        if not url_str and fichier_nom:
            url_str = url_by_file.get(fichier_nom, "")

        resume = to_str(row.get("resume", ""))
        idees = to_str(row.get("idees", ""))
        themes = to_str(row.get("themes", ""))

        youtube_id = extract_youtube_id(url_str)

        # 🚫 Masquer les cartes totalement vides
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
                st.markdown(f"### [{video_name}]({url_str})")
            else:
                st.markdown(f"### {video_name}")
            st.markdown(f"🗓️ *{video_date}*")
            if resume:
                st.markdown(f"📜 {resume}")


            # --- remplace TOUT votre bloc actuel "Voir Newsletter" par celui-ci ---
if fichier_nom:
    state_key = f"show_newsletter_{fichier_nom}"

    # Un seul bouton qui toggle l’affichage
    if st.button("📬 Newsletter liée à cette vidéo", key=f"btn_nl_{fichier_nom}"):
        toggle(state_key)

    if st.session_state.get(state_key, False):
        newsletter_contenu = charger_newsletter_html(fichier_nom)
        if newsletter_contenu:
            newsletter_contenu = fix_newsletter_html(newsletter_contenu)
            with st.expander("📬 Newsletter (ouvrir/fermer)", expanded=True):
                st.markdown(newsletter_contenu, unsafe_allow_html=True)
                bouton_telecharger_newsletter(fichier_nom, newsletter_contenu)
        else:
            st.warning("❌ Pas de newsletter disponible pour cette vidéo.")


            if themes:
                tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
                for theme in themes.split("|"):
                    t = theme.strip()
                    if t:
                        tags_html += "<span style='background-color: #D0E8FF; padding: 6px 12px; border-radius: 20px;'>{}</span>".format(t)
                tags_html += "</div>"
                st.markdown(tags_html, unsafe_allow_html=True)

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
            # Charger FAISS
            vectordb = FAISS.load_local(
                "faiss_transcripts",
                OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")),
                allow_dangerous_deserialization=True
            )

            # Recherche dans FAISS
            docs = vectordb.similarity_search(user_question, k=5)

            # Contexte pour GPT
            context = ""
            for doc in docs:
                url = doc.metadata.get("url", "URL inconnue")
                context += f"[Source: {url}]\n{doc.page_content}\n\n"

            # Construire prompt
            prompt = f"""
Tu es un expert de notre entreprise. Voici des extraits de nos formations :

{context}

Réponds précisément à la question suivante en utilisant uniquement ces extraits.
Si aucune information n'existe, réponds : "Je n'ai pas trouvé cette information dans notre base actuelle."

Question : {user_question}
"""

            # Appel à GPT-4 Turbo (ou modèle dispo)
            llm = ChatOpenAI(
                model="gpt-4-0125-preview",
                temperature=0.2,
                openai_api_key=openai.api_key
            )
            response = llm.invoke(prompt)

            st.success(response.content)
