
import os
import json
import whisper
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Paramètres
dossier_videos = Path("videos")
dossier_transcriptions = Path("transcriptions")
dossier_transcriptions.mkdir(exist_ok=True)
duree_bloc = 30  # en secondes

# Charger les URLs
df_urls = pd.read_csv("urls.csv")

def seconds_to_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def decouper_en_blocs(transcription, video_id, duree_bloc):
    segments = transcription["segments"]
    blocs = []
    bloc = {"start": segments[0]["start"], "text": "", "video_id": video_id}

    for seg in segments:
        if seg["start"] - bloc["start"] >= duree_bloc:
            bloc["text"] = bloc["text"].strip()
            blocs.append(bloc)
            bloc = {"start": seg["start"], "text": "", "video_id": video_id}
        bloc["text"] += " " + seg["text"]

    bloc["text"] = bloc["text"].strip()
    blocs.append(bloc)
    return blocs

all_blocs = []
model = whisper.load_model("medium")

for _, row in df_urls.iterrows():
    fichier = row["fichier"]
    url = row["url"]
    video_id = url.split("/")[-1]

    chemin_video = dossier_videos / fichier
    chemin_json = dossier_transcriptions / f"{video_id}.json"

    if not chemin_json.exists():
        print(f"🔁 Transcription de {fichier}...")
        result = model.transcribe(str(chemin_video), fp16=False)
        with open(chemin_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(f"✅ Transcription déjà existante pour {fichier}")

    with open(chemin_json, "r", encoding="utf-8") as f:
        transcription = json.load(f)
    blocs = decouper_en_blocs(transcription, video_id, duree_bloc)
    all_blocs.extend(blocs)

df_blocs = pd.DataFrame(all_blocs)
df_blocs.to_csv("blocs_fusionnes.csv", index=False)
print("✅ Fichier blocs_fusionnes.csv généré avec succès.")
