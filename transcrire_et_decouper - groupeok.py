import torch
import whisper
import os
from pathlib import Path
from tqdm import tqdm
import json
import time

# === CONFIGURATION ===
DOSSIER_VIDEOS = "videos"
DOSSIER_JSON = "json"
DUREE_BLOC_SECONDES = 30
TIMEOUT_PAR_VIDEO = 1200  # ⏱️ 20 minutes max par vidéo

# Préparation des dossiers
os.makedirs(DOSSIER_JSON, exist_ok=True)

# Charger Whisper
print("🔁 Chargement du modèle Whisper...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=DEVICE)
print(f"✅ Modèle chargé sur : {DEVICE}")

# Récupérer les vidéos
videos = list(Path(DOSSIER_VIDEOS).glob("*.mp4"))
print("🎞️ Vidéos trouvées :", [v.name for v in videos])

# Transcription vidéo par vidéo
for chemin_video in tqdm(videos, desc="📼 Transcription en cours"):
    nom_base = chemin_video.stem
    json_sortie = Path(DOSSIER_JSON) / f"{nom_base}.json"
    
    if json_sortie.exists():
        print(f"✅ Déjà traité : {json_sortie.name}")
        continue

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
