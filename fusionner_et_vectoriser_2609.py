import os
import glob
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- OpenAI SDK (nouvelle version) ---
from openai import OpenAI
client = OpenAI()  # lit automatiquement OPENAI_API_KEY

# 📚 Charger tous les blocs CSV depuis le dossier blocs/
bloc_files = glob.glob("blocs/*.csv")

dfs = []
for f in bloc_files:
    df = pd.read_csv(f)
    # Ajouter une colonne 'fichier' basée sur le nom de fichier (comme avant)
    video_name = Path(f).stem.replace("_blocs", "")
    df["fichier"] = video_name
    dfs.append(df)

# Fusionner
blocs_fusionnes = pd.concat(dfs, ignore_index=True)

# 🔗 Charger la table des URLs (identique: encodage cp1252)
urls_df = pd.read_csv("urls.csv", encoding="cp1252")
urls_dict = dict(zip(urls_df["fichier"], urls_df["url"]))

# Ajouter la colonne 'url' correspondante
blocs_fusionnes["url"] = blocs_fusionnes["fichier"].map(urls_dict)

# Ne garder que les colonnes utiles (comme avant)
blocs_fusionnes = blocs_fusionnes[["start", "end", "text", "url"]]

# Sauvegarde du CSV final (même comportement que l'original : pas d'encoding explicite)
blocs_fusionnes.to_csv("blocs_fusionnes.csv", index=False)

print(f"✅ {len(blocs_fusionnes)} blocs fusionnés et enrichis !")

# --- Embeddings ---
def embed_openai(batch_texts):
    # Appel minimaliste pour rester proche de l’original
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch_texts
    )
    return [d.embedding for d in resp.data]

# Découper en batchs (valeur d'origine : 1000)
BATCH_SIZE = 1000
vectors = []

for i in tqdm(range(0, len(blocs_fusionnes), BATCH_SIZE), desc="🔍 Vectorisation"):
    batch_texts = blocs_fusionnes["text"].iloc[i:i+BATCH_SIZE].fillna("").astype(str).tolist()
    batch_vectors = embed_openai(batch_texts)
    vectors.extend(batch_vectors)

# Sauvegarde du vecteur
with open("vecteurs.pkl", "wb") as f:
    pickle.dump(vectors, f)

print("✅ Vectorisation terminée et vecteurs enregistrés dans vecteurs.pkl")
