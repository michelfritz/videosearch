
import whisper
import json
import os
import csv
from pathlib import Path
from tqdm import tqdm

DOSSIER_VIDEOS = Path("videos")
DOSSIER_SORTIE = Path("resultats")
DOSSIER_SORTIE.mkdir(exist_ok=True)
DUREE_BLOC = 30  # en secondes

model = whisper.load_model("base", device="cuda" if whisper.torch.cuda.is_available() else "cpu")

def decouper_blocs(segments, duree_bloc, suffixe_video):
    blocs = []
    bloc = {"start": 0, "text": "", "video": suffixe_video}
    for segment in segments:
        if segment["start"] >= bloc["start"] + duree_bloc:
            blocs.append(bloc)
            bloc = {"start": segment["start"], "text": segment["text"], "video": suffixe_video}
        else:
            bloc["text"] += " " + segment["text"]
    blocs.append(bloc)
    return blocs

toutes_les_lignes = []

videos = list(DOSSIER_VIDEOS.glob("*.mp4"))

for chemin_video in tqdm(videos, desc="📼 Transcription en cours"):
    try:
        print(f"🔊 Transcription : {chemin_video.name}")
        result = model.transcribe(str(chemin_video), fp16=False)
        nom_json = DOSSIER_SORTIE / f"{chemin_video.stem}.json"
        with open(nom_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        suffixe = chemin_video.stem
        blocs = decouper_blocs(result["segments"], DUREE_BLOC, suffixe)
        for bloc in blocs:
            toutes_les_lignes.append([bloc["video"], bloc["start"], bloc["text"]])

    except Exception as e:
        print(f"❌ Erreur sur {chemin_video.name} : {e}")
        continue

with open(DOSSIER_SORTIE / "blocs_fusionnes.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "start", "text"])
    writer.writerows(toutes_les_lignes)

print("✅ Transcription et découpage terminés.")
