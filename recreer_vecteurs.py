import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Chargement des blocs
df = pd.read_csv("blocs_transcription.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Nettoyage et vectorisation
textes = df["text"].fillna("").astype(str).tolist()
vecteurs = model.encode(textes, convert_to_numpy=True)

# Sauvegarde propre en .pkl
with open("vecteurs.pkl", "wb") as f:
    pickle.dump(vecteurs, f)

print("✅ vecteurs.pkl recréé avec succès !")