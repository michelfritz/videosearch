
import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import whisper
import openai
from openai import OpenAIError

# Configurations
DOSSIER_VIDEOS = "videos"
DOSSIER_TRANSCRIPTIONS = "transcriptions"
DOSSIER_BLOCS = "blocs"
FICHIER_URLS = "urls.csv"
FICHIER_BLOCS_FUSIONNES = "blocs_fusionnes.csv"
FICHIER_VECTEURS = "vecteurs.pkl"
MODEL_WHISPER = "base"  # base ou small
DUREE_BLOC_SECONDES = 30

# Chargement de la clé OpenAI depuis les variables d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Clé API OpenAI non trouvée. Assurez-vous d'avoir défini la variable d'environnement OPENAI_API_KEY.")
openai.api_key = OPENAI_API_KEY

# Préparation des dossiers
os.makedirs(DOSSIER_TRANSCRIPTIONS, exist_ok=True)
os.makedirs(DOSSIER_BLOCS, exist_ok=True)

# Détection des vidéos
videos = list(Path(DOSSIER_VIDEOS).glob("*.mp4"))
print(f"🎥 {len(videos)} vidéo(s) détectée(s) dans {DOSSIER_VIDEOS}/")

# Chargement du modèle Whisper
print("🔁 Chargement du modèle Whisper...")
model = whisper.load_model(MODEL_WHISPER)
DEVICE = "cuda" if whisper.torch.cuda.is_available() else "cpu"
print(f"✅ Modèle chargé sur : {DEVICE}")

# Lecture du fichier urls.csv
urls_df = pd.read_csv(FICHIER_URLS)
urls_mapping = dict(zip(urls_df["fichier"], urls_df["url"]))

# Transcription et découpage
for chemin_video in tqdm(videos, desc="📼 Traitement des vidéos"):
    nom = chemin_video.stem
    transcription_path = Path(DOSSIER_TRANSCRIPTIONS) / f"{nom}.json"
    blocs_path = Path(DOSSIER_BLOCS) / f"{nom}_blocs.csv"

    if transcription_path.exists():
        print(f"📝 JSON existant pour {nom}, saut transcription.")
    else:
        print(f"🔊 Transcription : {nom}")
        result = model.transcribe(str(chemin_video), fp16=(DEVICE=="cuda"), language="fr")
        with open(transcription_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Découper en blocs
    with open(transcription_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    segments = result.get("segments", [])
    blocs = []
    texte_bloc = ""
    debut_bloc = None

    for segment in segments:
        start, end, text = segment["start"], segment["end"], segment["text"]

        if debut_bloc is None:
            debut_bloc = start

        texte_bloc += " " + text.strip()

        if end - debut_bloc >= DUREE_BLOC_SECONDES:
            blocs.append({"start": debut_bloc, "end": end, "text": texte_bloc.strip(), "fichier": nom})
            debut_bloc = None
            texte_bloc = ""

    if texte_bloc:
        blocs.append({"start": debut_bloc, "end": end, "text": texte_bloc.strip(), "fichier": nom})

    df_blocs = pd.DataFrame(blocs)
    df_blocs.to_csv(blocs_path, index=False, encoding="utf-8-sig")
    print(f"✅ {len(blocs)} blocs exportés pour {nom}.")

# Fusionner tous les blocs
print("🔗 Fusion des blocs...")
blocs_fusionnes = pd.concat([pd.read_csv(f) for f in Path(DOSSIER_BLOCS).glob("*_blocs.csv")], ignore_index=True)

# Ajouter colonne youtube_id
blocs_fusionnes["url"] = blocs_fusionnes["fichier"].map(urls_mapping)
blocs_fusionnes = blocs_fusionnes[["start", "end", "text", "url", "fichier"]]
blocs_fusionnes.to_csv(FICHIER_BLOCS_FUSIONNES, index=False, encoding="utf-8-sig")
print(f"✅ {len(blocs_fusionnes)} blocs fusionnés.")

# Vectorisation avec OpenAI
print("🧠 Vectorisation des blocs...")
embeddings = []
texts = blocs_fusionnes["text"].tolist()

BATCH_SIZE = 100
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🧪 Envoi à OpenAI"):
    batch = texts[i:i+BATCH_SIZE]
    try:
        response = openai.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [r.embedding for r in response.data]
        embeddings.extend(batch_embeddings)
    except OpenAIError as e:
        print(f"Erreur lors de l'envoi à OpenAI : {e}")
        raise e

vecteurs = np.array(embeddings)
with open(FICHIER_VECTEURS, "wb") as f:
    pickle.dump(vecteurs, f)

print(f"✅ Vectorisation terminée ({vecteurs.shape[0]} vecteurs).")
print("🏁 Script terminé avec succès.")
