import torch
import whisper
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# === CONFIGURATION ===
DOSSIER_VIDEOS = "videos"
DOSSIER_JSON = "json"
DOSSIER_BLOCS = "blocs"
DUREE_BLOC_SECONDES = 30
TIMEOUT_PAR_VIDEO = 1200  # 20 minutes max par vidéo

# Préparation des dossiers
os.makedirs(DOSSIER_JSON, exist_ok=True)
os.makedirs(DOSSIER_BLOCS, exist_ok=True)

# === CHARGER WHISPER ===
print("🔁 Chargement du modèle Whisper...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=DEVICE)
print(f"✅ Modèle chargé sur : {DEVICE}")

# === RECUPERER LES VIDEOS ===
videos = list(Path(DOSSIER_VIDEOS).glob("*.mp4"))
print("🎞️ Vidéos trouvées :", [v.name for v in videos])

# === TRANSCRIRE ET DECOUPER ===
for chemin_video in tqdm(videos, desc="📼 Transcription en cours"):
    nom_base = chemin_video.stem
    json_sortie = Path(DOSSIER_JSON) / f"{nom_base}.json"
    csv_sortie = Path(DOSSIER_BLOCS) / f"{nom_base}_blocs.csv"
    
    if csv_sortie.exists():
        print(f"✅ Blocs déjà générés pour : {nom_base}")
        continue

    if not json_sortie.exists():
        print(f"🔊 Transcription : {chemin_video.name}")
        try:
            start_time = time.time()
            result = model.transcribe(str(chemin_video), language="fr", verbose=False, fp16=(DEVICE=="cuda"))
            elapsed = time.time() - start_time

            if elapsed > TIMEOUT_PAR_VIDEO:
                print(f"⚠️ Temps dépassé pour {chemin_video.name}, ignorée.")
                continue

            with open(json_sortie, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Sauvegardé : {json_sortie.name} en {round(elapsed/60, 1)} min")

        except KeyboardInterrupt:
            print("⛔ Interruption manuelle 🖐️")
            break
        except Exception as e:
            print(f"❌ Erreur sur {chemin_video.name} : {e}")
            continue
    else:
        print(f"📝 JSON déjà existant pour {nom_base}, saut transcription.")

    # Découper le JSON en blocs de 30 secondes
    try:
        with open(json_sortie, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        blocs = []
        bloc = {"start": None, "end": None, "text": ""}

        for seg in segments:
            if bloc["start"] is None:
                bloc["start"] = seg["start"]
            bloc["end"] = seg["end"]
            bloc["text"] += " " + seg["text"].strip()

            if bloc["end"] - bloc["start"] >= DUREE_BLOC_SECONDES:
                blocs.append(bloc)
                bloc = {"start": None, "end": None, "text": ""}

        if bloc["text"].strip():
            blocs.append(bloc)

        df_blocs = pd.DataFrame(blocs)
        df_blocs.to_csv(csv_sortie, index=False, encoding="utf-8")
        print(f"✅ {len(blocs)} blocs exportés pour {nom_base}.")

    except Exception as e:
        print(f"❌ Erreur de découpe pour {nom_base} : {e}")

print("🏁 Traitement terminé pour toutes les vidéos.")
