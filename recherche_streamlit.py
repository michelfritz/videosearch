import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

# Configuration Streamlit
st.set_page_config(page_title="Recherche IA dans vidéos", layout="wide")

# Charger la clé API OpenAI depuis les variables d'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔧 Chargement modèle OpenAI pour embedding
@st.cache_resource
def embed_openai(texts):
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return np.array([d.embedding for d in response.data])

# 📂 Chargement du fichier de blocs et vecteurs
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

# 🔎 Fonction recherche
def rechercher_similaires(vecteur_query, vecteurs, seuil=0.4, top_k=5):
    vecteur_query = vecteur_query.squeeze()
    similarities = np.dot(vecteurs, vecteur_query)
    indices = np.where(similarities >= seuil)[0]
    top_indices = indices[np.argsort(similarities[indices])[::-1][:top_k]]
    return top_indices, similarities[top_indices]

# 🖼 Interface utilisateur
st.title("🔍 Recherche intelligente dans vos vidéos YouTube 📽")

query = st.text_input("🧠 Que veux-tu savoir ?", "")

seuil = st.slider("🎯 Ajuste le seuil de pertinence (plus haut = plus strict)", 0.2, 0.95, 0.4, 0.01)

if query:
    with st.spinner("⏳ Chargement du modèle et des données..."):
        df, vecteurs = charger_donnees()
        vecteur_query = embed_openai([query])
        indices, scores = rechercher_similaires(vecteur_query, vecteurs, seuil)

    if len(indices) == 0:
        st.warning("❌ Aucun résultat trouvé avec ce seuil.")
    else:
        st.markdown("### 📋 Résultats pertinents :")
        for idx, score in zip(indices, scores):
            bloc = df.iloc[idx]
            start_time = int(float(bloc["start"]))
            url_complet = bloc["url"]

            if "watch?v=" in url_complet:
                youtube_id = url_complet.split("watch?v=")[-1].split("&")[0]
            elif "youtu.be/" in url_complet:
                youtube_id = url_complet.split("youtu.be/")[-1].split("?")[0]
            else:
                youtube_id = ""

            video_url = f"https://www.youtube.com/embed/{youtube_id}?start={start_time}&autoplay=0"

            with st.expander(f"⏱️ {start_time}s — Score {round(score, 2)} — 💬 {bloc['text'][:60]}..."):
                st.markdown(f"**Texte complet :** {bloc['text']}")
                st.components.v1.iframe(video_url, height=315)
