import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")

# Afficher le logo
st.image("logo_lucarne.png", width=180)
st.markdown("# 📚 Base de connaissance A LA LUCARNE")

# Clé API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Charger les données
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls_et_idees_themes():
    urls = pd.read_csv("urls.csv", encoding="utf-8")
    urls["titre"] = urls["titre"].fillna("Titre inconnu")
    urls["date"] = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")

    idees = pd.read_csv("idees.csv", encoding="utf-8")
    idees["idees"] = idees["idees"].fillna("")

    idees_v2 = pd.read_csv("idees_v2.csv", encoding="utf-8")

    themes = pd.read_csv("themes.csv", encoding="utf-8")
    themes["themes"] = themes["themes"].fillna("")

    mesthemes = pd.read_csv("mesthemes.csv", encoding="utf-8")
    mesthemes_list = mesthemes["themes"].dropna().tolist()

    df = pd.merge(urls, idees, on="fichier", how="left")
    df = pd.merge(df, themes, on="fichier", how="left")
    return df, idees_v2, themes, mesthemes_list

# Fonction OpenAI

def embed_openai(query):
    response = openai.embeddings.create(
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

# Interface principale
df, vecteurs = charger_donnees()
urls_df, idees_v2_df, themes_df, mesthemes_list = charger_urls_et_idees_themes()

# Préparer les thèmes
all_themes = set()
for theme_list in themes_df["themes"].dropna():
    for theme in theme_list.split("|"):
        theme = theme.strip()
        if theme:
            all_themes.add(theme)

if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = ""

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

menu = st.sidebar.radio("Navigation", ["🔍 Recherche", "🎥 Toutes les vidéos"])

if menu == "🔍 Recherche":
    col1, col2 = st.columns([3,1])

    with col1:
        st.text_input("🔍 Que veux-tu savoir ?", key="search_query")
    with col2:
        if st.button("🔄 Réinitialiser"):
            st.session_state.selected_theme = ""
            st.session_state.search_query = ""
            st.experimental_rerun()

    seuil = st.slider("🌟 Exigence des résultats", 0.1, 0.9, 0.5, 0.05)

    # Mes thèmes personnalisés
    with st.expander("✨ Mes Thèmes personnalisés", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(mesthemes_list)):
            if cols[i % 4].button(theme, key=f"mestheme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.search_query = ""
                st.experimental_rerun()

    # Tous les thèmes
    with st.expander("🏷️ Tous les Thèmes", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(all_themes)):
            if cols[i % 4].button(theme, key=f"theme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.search_query = ""
                st.experimental_rerun()

    query = st.session_state.search_query.strip() if st.session_state.search_query.strip() else st.session_state.selected_theme

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
                url_complet = bloc["url"]
                youtube_id = ""
                if "watch?v=" in url_complet:
                    youtube_id = url_complet.split("watch?v=")[-1]
                elif "youtu.be/" in url_complet:
                    youtube_id = url_complet.split("youtu.be/")[-1]

                start_time = int(float(bloc["start"]))
                embed_url = f"https://www.youtube.com/embed/{youtube_id}?start={start_time}&autoplay=0"

                with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}... (score: {score:.2f})"):
                    st.markdown(f"**Texte complet :** {bloc['text']}")
                    if youtube_id:
                        st.components.v1.iframe(embed_url, height=315)

elif menu == "🎥 Toutes les vidéos":
    # (Le reste du code reste inchangé)
