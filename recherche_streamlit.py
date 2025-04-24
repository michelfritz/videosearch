import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche IA dans transcription", layout="wide")

# Chargement du modèle SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Chargement du CSV et du fichier vecteurs
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_transcription.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

# Embedding de la requête
def embed(texts, model):
    return model.encode(texts, convert_to_numpy=True)

# Recherche par similarité
def rechercher_similaires(vecteur_query, vecteurs, seuil=0.6, top_k=10):
    vecteur_query = vecteur_query.squeeze()
    similarities = np.dot(vecteurs, vecteur_query)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    résultats_filtrés = [(i, similarities[i]) for i in top_k_indices if similarities[i] >= seuil]
    return résultats_filtrés

# Interface
st.title("🔍 Recherche intelligente dans la transcription")

query = st.text_input("🧠 Que veux-tu savoir ?", "")
seuil = st.slider("🎚️ Seuil de précision (plus élevé = plus exigeant)", 0.0, 1.0, 0.6, 0.01)

if query:
    with st.spinner("Chargement du modèle et des données..."):
        model = load_model()
        df, vecteurs = charger_donnees()
        vecteur_query = embed([query], model)
        résultats = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)
