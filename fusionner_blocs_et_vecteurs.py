
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import uuid

# --- CONFIG ---
DOSSIER_BLOCS = Path("blocs")
FICHIER_URLS = Path("urls.csv")
FICHIER_FUSION = Path("blocs_fusionnes.csv")
FICHIER_VECTEURS = Path("vecteurs.pkl")
MODELE_OPENAI = "text-embedding-3-small"

# --- INITIALISATION ---
client = OpenAI()
tqdm.pandas()

def embed_openai_batch(texts):
    response = client.embeddings.create(
        input=texts,
        model=MODELE_OPENAI,
        encoding_format="float"
    )
    return [np.array(d.embedding, dtype=np.float32) for d in response.data]

print("📂 Chargement des fichiers de blocs...")
fichiers_csv = sorted(DOSSIER_BLOCS.glob("*.csv"))
df_all = []

for f in fichiers_csv:
    df = pd.read_csv(f)
    nom_video = f.stem.lower()
    df["nom_fichier"] = nom_video + ".mp4"
    df["bloc_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df_all.append(df)

df_fusion = pd.concat(df_all, ignore_index=True)

print("🔗 Ajout des URLs à partir de urls.csv...")
urls_df = pd.read_csv(FICHIER_URLS)
df_fusion = df_fusion.merge(urls_df, left_on="nom_fichier", right_on="fichier", how="left")

if df_fusion["url"].isnull().any():
    print("❗ Certaines vidéos n'ont pas d'URL associée dans urls.csv")

# Optionnel : Nettoyage
df_fusion = df_fusion.drop(columns=["fichier"])
df_fusion = df_fusion[["bloc_id", "nom_fichier", "start", "text", "url"]]  # réorganise

print("💾 Sauvegarde du fichier blocs_fusionnes.csv")
df_fusion.to_csv(FICHIER_FUSION, index=False)

print("🧠 Vectorisation avec OpenAI...")
embeddings = []
for i in tqdm(range(0, len(df_fusion), 50)):
    batch = df_fusion["text"].iloc[i:i+50].fillna("").astype(str).tolist()
    embeddings.extend(embed_openai_batch(batch))

embeddings = np.vstack(embeddings)

print("💾 Sauvegarde de vecteurs.pkl")
with open(FICHIER_VECTEURS, "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Fusion + vectorisation terminée.")
