import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

st.set_page_config(page_title="Recherche IA dans transcriptions", layout="wide")

# 🔐 Clé API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📚 Charger les données
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls():
    try:
        urls = pd.read_csv("urls.csv", encoding="utf-8")
    except UnicodeDecodeError:
        urls = pd.read_csv("urls.csv", encoding="cp1252")
    urls["titre"] = urls["titre"].fillna("Titre inconnu")
    urls["date"] = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")
    return urls

# 🔎 Embedding OpenAI
def embed_openai(query):
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small",
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

# 🔥 Recherche de similarité
def rechercher_similaires(vecteur_query, vecteurs, top_k=5, seuil=0.3):
    similarities = np.dot(vecteurs, vecteur_query)
    indices = np.where(similarities >= seuil)[0]
    top_indices = indices[np.argsort(similarities[indices])[::-1][:top_k]]
    return top_indices, similarities[top_indices]

# 🛠 Interface Streamlit
st.title("🔍 Recherche intelligente dans les transcriptions")

# 📚 Charger les données
df, vecteurs = charger_donnees()
urls_df = charger_urls()

# 📂 Menu latéral
menu = st.sidebar.radio("Navigation", ["🔍 Recherche", "🎥 Toutes les vidéos"])

if menu == "🔍 Recherche":
    query = st.text_input("🧐 Que veux-tu savoir ?", "")
    seuil = st.slider("🎯 Exigence des résultats (plus haut = plus précis)", 0.1, 0.9, 0.5, 0.05)

    if query:
        with st.spinner("🔍 Recherche en cours..."):
            vecteur_query = embed_openai(query)
            indices, scores = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)

        if len(indices) == 0:
            st.warning("Aucun résultat trouvé. 😕 Essaie une autre requête ou baisse l'exigence.")
        else:
            st.markdown("### 🌟 Résultats pertinents :")
            for idx, score in zip(indices, scores):
                bloc = df.iloc[idx]
                url_complet = bloc["url"]
                if "watch?v=" in url_complet:
                    youtube_id = url_complet.split("watch?v=")[-1]
                elif "youtu.be/" in url_complet:
                    youtube_id = url_complet.split("youtu.be/")[-1]
                else:
                    youtube_id = ""

                start_time = int(float(bloc["start"]))
                embed_url = f"https://www.youtube.com/embed/{youtube_id}?start={start_time}&autoplay=0"

                with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}... (score: {score:.2f})"):
                    st.markdown(f"**Texte complet :** {bloc['text']}")
                    if youtube_id:
                        st.components.v1.iframe(embed_url, height=315)

elif menu == "🎥 Toutes les vidéos":
    st.header("📚 Liste des vidéos disponibles")

    recherche = st.text_input("🔎 Recherche par titre ou résumé", "")

    tri = st.selectbox(
        "📜 Trier par",
        ("Date récente", "Date ancienne", "Titre A → Z", "Titre Z → A")
    )

    if recherche:
        urls_df = urls_df[urls_df["titre"].str.contains(recherche, case=False, na=False) | urls_df["resume"].str.contains(recherche, case=False, na=False)]

    if tri == "Date récente":
        urls_df = urls_df.sort_values("date", ascending=False)
    elif tri == "Date ancienne":
        urls_df = urls_df.sort_values("date", ascending=True)
    elif tri == "Titre A → Z":
        urls_df = urls_df.sort_values("titre", ascending=True)
    elif tri == "Titre Z → A":
        urls_df = urls_df.sort_values("titre", ascending=False)

    st.markdown(f"### 🎬 {len(urls_df)} vidéo(s) trouvée(s)")

    for _, row in urls_df.iterrows():
        video_name = row.get("titre", "Titre inconnu")
        video_date = row.get("date", "Date inconnue")
        url_complet = row["url"]
        resume = row.get("resume", "")

        if "watch?v=" in url_complet:
            youtube_id = url_complet.split("watch?v=")[-1]
        elif "youtu.be/" in url_complet:
            youtube_id = url_complet.split("youtu.be/")[-1]
        else:
            youtube_id = ""

        thumbnail_url = f"https://img.youtube.com/vi/{youtube_id}/0.jpg"

        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(thumbnail_url, width=140)
        with col2:
            st.markdown(f"### [{video_name}]({url_complet})")
            st.markdown(f"🗓️ *{video_date}*")
            if resume:
                st.markdown(f"📜 {resume}")
            st.markdown(f"[▶️ Voir sur YouTube]({url_complet})")
        st.markdown("---")
