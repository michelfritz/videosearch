import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche IA dans transcription", layout="wide")

# Cache les ressources
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_transcription.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

# Embedding
def embed(texts, model):
    return model.encode(texts, convert_to_numpy=True)

# Similarité cosinus
def rechercher_similaires(vecteur_query, vecteurs, top_k=5):
    vecteur_query = vecteur_query.squeeze()
    similarities = np.dot(vecteurs, vecteur_query)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return top_k_indices, similarities[top_k_indices]

# Interface Streamlit
st.title("🔍 Recherche intelligente dans la transcription")

query = st.text_input("🧠 Que veux-tu savoir ?", "")

if query:
    with st.spinner("Chargement du modèle et des données..."):
        model = load_model()
        df, vecteurs = charger_donnees()
        vecteur_query = embed([query], model)
        indices, scores = rechercher_similaires(vecteur_query, vecteurs)

    st.markdown("### 🎯 Résultats pertinents :")
    for idx, score in zip(indices, scores):
        bloc = df.iloc[idx]
        start_time = int(float(bloc["start"]))
        video_url = f"https://www.youtube.com/embed/t21LM4CXaqE?start={start_time}&autoplay=0"

        with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}..."):
            st.markdown(f"**Texte complet :** {bloc['text']}")
            st.components.v1.iframe(video_url, height=315)
