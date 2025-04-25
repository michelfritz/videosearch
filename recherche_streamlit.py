import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

# 🔐 Clé API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Recherche IA multi-vidéos", layout="wide")

# Embedding via OpenAI
def embed_openai(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

# Similarité cosinus
def rechercher_similaires(vecteur_query, vecteurs, top_k=5):
    vecteur_query = vecteur_query.squeeze()
    similarities = np.dot(vecteurs, vecteur_query)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return top_k_indices, similarities[top_k_indices]

# Interface
st.title("🔎 Recherche intelligente dans plusieurs vidéos")

query = st.text_input("🧠 Que veux-tu savoir ?", "")
top_k = st.slider("🔢 Nombre de résultats", 1, 10, 5)

if query:
    with st.spinner("🔍 Traitement de ta question..."):
        df, vecteurs = charger_donnees()
        vecteur_query = embed_openai(query)
        indices, scores = rechercher_similaires(vecteur_query, vecteurs, top_k=top_k)

    st.markdown("### 🎯 Résultats pertinents :")
    for idx, score in zip(indices, scores):
        bloc = df.iloc[idx]
        start_time = int(float(bloc["start"]))
        url_complet = bloc["url"]

        # Extraire l'ID de la vidéo à partir de l'URL
        if "watch?v=" in url_complet:
            video_id = url_complet.split("watch?v=")[1]
        elif "youtu.be/" in url_complet:
            video_id = url_complet.split("youtu.be/")[1]
        else:
            video_id = url_complet  # fallback
        
        # Construction de l'embed YouTube avec démarrage au bon moment
        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_time}&autoplay=0"

        with st.expander(f"🎥 {video_id} ⏱️ {start_time}s — 💬 {bloc['text'][:60]}..."):
            st.markdown(f"**Texte complet :** {bloc['text']}")
            st.components.v1.iframe(embed_url, height=315)
