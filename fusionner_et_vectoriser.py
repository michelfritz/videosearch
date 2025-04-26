import pandas as pd
import pickle
import openai
import os
from tqdm import tqdm
import glob
from pathlib import Path  # ✅ Ajout important

# 🔑 Chargement de ta clé API OpenAI depuis l'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📚 Charger tous les blocs CSV depuis le dossier blocs/
bloc_files = glob.glob("blocs/*.csv")

dfs = []
for f in bloc_files:
    df = pd.read_csv(f)
    # Ajouter une colonne pour le nom de la vidéo
    video_name = Path(f).stem.replace("_blocs", "")
    df["fichier"] = video_name
    dfs.append(df)

# Fusionner tous les blocs
blocs_fusionnes = pd.concat(dfs, ignore_index=True)

# 🌐 Charger ton vrai urls.csv (avec 'fichier' et 'url')
urls_df = pd.read_csv("urls.csv", encoding="cp1252")


# 🔥 Correction ici : utiliser 'url' au lieu de 'youtube_id'
urls_dict = dict(zip(urls_df["fichier"], urls_df["url"]))

# Ajouter la colonne 'url' correspondante
blocs_fusionnes["url"] = blocs_fusionnes["fichier"].map(urls_dict)

# Ne garder que les colonnes utiles
blocs_fusionnes = blocs_fusionnes[["start", "end", "text", "url"]]

# Sauvegarde du CSV final
blocs_fusionnes.to_csv("blocs_fusionnes.csv", index=False)

print(f"✅ {len(blocs_fusionnes)} blocs fusionnés et enrichis !")

# 🧠 Vectorisation avec OpenAI
def embed_openai(texts):
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [d.embedding for d in response.data]

# Découper en batchs si besoin
BATCH_SIZE = 1000
vectors = []

for i in tqdm(range(0, len(blocs_fusionnes), BATCH_SIZE), desc="🔍 Vectorisation"):
    batch_texts = blocs_fusionnes["text"].iloc[i:i+BATCH_SIZE].tolist()
    batch_vectors = embed_openai(batch_texts)
    vectors.extend(batch_vectors)

# Sauvegarde du vecteur
with open("vecteurs.pkl", "wb") as f:
    pickle.dump(vectors, f)

print("✅ Vectorisation terminée et vecteurs enregistrés dans vecteurs.pkl")
