import pandas as pd
import numpy as np
import pickle
import openai
import os
from tqdm import tqdm

# 🔐 Soit tu mets ta clé ici, soit tu la configures dans ton environnement
openai.api_key = os.getenv("OPENAI_API_KEY")  # Option recommandée
# openai.api_key = "sk-..."  # Option alternative : à éviter de hardcoder

# 🔹 Chargement du fichier CSV
df = pd.read_csv("blocs_transcription.csv")

# Vérification des colonnes attendues
if "text" not in df.columns:
    raise ValueError("La colonne 'text' est manquante dans le fichier CSV.")
df["text"] = df["text"].fillna("").astype(str)

# 🔹 Fonction pour interroger l'API OpenAI avec batching
def embed_openai_batch(texts, model="text-embedding-3-small", batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Vectorisation"):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# 🔹 Création des vecteurs
textes = df["text"].tolist()
vecteurs = embed_openai_batch(textes)

# 🔹 Sauvegarde
with open("vecteurs_openai.pkl", "wb") as f:
    pickle.dump(vecteurs, f)

print("✅ Vecteurs vectorisés et sauvegardés dans vecteurs_openai.pkl")
