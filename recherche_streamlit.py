import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

st.set_page_config(page_title="Recherche IA dans transcription", layout="wide")

# 🔐 Clé API OpenAI (à mettre en variable d’environnement dans Streamlit Cloud aussi !)
openai.api_key = os.getenv("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")

@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_transcription.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, np.array(vecteurs)

def embed_openai(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def rechercher_similaires(vecteur_query, vecteurs, top_k=5):
    similarities = np.dot(vecteurs, vecteur_query)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return top_k_indices, similarities[top_k_indices]

# 🔍 UI
st.title("🔍 Recherche intelligente dans la transcription")

score_threshold = st.slider("Filtrer les résultats par score de similarité", 0.0, 1.0, 0.7, 0.01)
query = st.text_input("🧠 Que veux-tu savoir ?", "")

if query:
    with st.spinner("Chargement..."):
        df, vecteurs = charger_donnees()
        vecteur_query = embed_openai(query)
        indices, scores = rechercher_similaires(vecteur_query, vecteurs)

    st.markdown("### 🎯 Résultats pertinents :")
    for idx, score in zip(indices, scores):
        if score < score_threshold:
            continue
        bloc = df.iloc[idx]
        start_time = int(float(bloc["start"]))
        video_url = f"https://www.youtube.com/embed/t21LM4CXaqE?start={start_time}&autoplay=0"
        with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}... (score: {score:.2f})"):
            st.markdown(f"**Texte complet :** {bloc['text']}")
            st.components.v1.iframe(video_url, height=315)
