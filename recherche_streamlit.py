import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche IA dans transcription", layout="wide")

# Chargement du modèle avec cache
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Chargement des données avec cache
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_transcription.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

# Fonction d'embedding
def embed(texts, model):
    return model.encode(texts, convert_to_numpy=True)

# Fonction de recherche par similarité cosinus
def rechercher_similaires(vecteur_query, vecteurs, top_k=5):
    vecteur_query = vecteur_query[0]  # Corrige la forme (shape)
    similarities = np.dot(vecteurs, vecteur_query)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return top_k_indices, similarities[top_k_indices]

# Interface utilisateur
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
        st.markdown(f"**⏱️ Timestamp**: `{bloc['debut']}`")
        st.markdown(f"**💬 Texte**: {bloc['texte']}")
        st.markdown("---")
