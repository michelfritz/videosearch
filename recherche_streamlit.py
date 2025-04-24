import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche IA dans transcription", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_transcription.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

def embed(texts, model):
    return model.encode(texts, convert_to_numpy=True)

def rechercher_similaires(vecteur_query, vecteurs, seuil=0.3, top_k=10):
    vecteur_query = vecteur_query.squeeze()
    similarities = np.dot(vecteurs, vecteur_query)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    résultats_filtrés = [(i, similarities[i]) for i in top_k_indices if similarities[i] >= seuil]
    score_max = similarities[top_k_indices[0]] if len(top_k_indices) > 0 else 0
    return résultats_filtrés, score_max

st.title("🔍 Recherche intelligente dans la transcription")

query = st.text_input("🧠 Que veux-tu savoir ?", "")
seuil = st.slider("🎚️ Seuil de précision (plus élevé = plus exigeant)", 0.0, 1.0, 0.3, 0.01)

if query:
    with st.spinner("Chargement du modèle et des données..."):
        model = load_model()
        df, vecteurs = charger_donnees()
        vecteur_query = embed([query], model)
        résultats, score_max = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)

    st.markdown(f"**🔎 Score de similarité maximum obtenu :** `{score_max:.2f}`")
    
    if not résultats:
        st.warning("Aucun résultat trouvé au-dessus du seuil. Diminue le seuil pour élargir la recherche.")
    else:
        st.markdown("### 🎯 Résultats pertinents :")
        for idx, score in résultats:
            bloc = df.iloc[idx]
            start_time = int(float(bloc["start"]))
            video_url = f"https://www.youtube.com/embed/t21LM4CXaqE?start={start_time}&autoplay=0"
            
            with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}..."):
                st.markdown(f"**🧠 Similarité :** `{score:.2f}`")
                st.markdown(f"**📝 Texte complet :** {bloc['text']}")
                st.components.v1.iframe(video_url, height=315)
