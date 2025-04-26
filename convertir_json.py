import os
import json
import pandas as pd
from pathlib import Path

# --- Dossiers ---
DOSSIER_JSON = "json"
DOSSIER_SRT = "srt"
DOSSIER_RESUME = "resume"
DOSSIER_BLOCS = "blocs"

# --- Créer les dossiers si pas existants ---
os.makedirs(DOSSIER_SRT, exist_ok=True)
os.makedirs(DOSSIER_RESUME, exist_ok=True)
os.makedirs(DOSSIER_BLOCS, exist_ok=True)

# --- Fonction pour convertir secondes en format SRT ---
def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# --- Traitement fichier par fichier ---
for fichier_json in Path(DOSSIER_JSON).glob("*.json"):
    nom = fichier_json.stem
    print(f"\n✨ Traitement de {nom}...")

    with open(fichier_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    if not segments:
        print(f"\u26a0\ufe0f Aucun segment trouvé dans {nom}.json. Fichier ignoré.")
        continue

    resegmented = []
    buffer_text = ""
    buffer_start = None
    buffer_end = None

    for seg in segments:
        if buffer_start is None:
            buffer_start = seg["start"]

        buffer_text += (" " if buffer_text else "") + seg["text"]
        buffer_end = seg["end"]

        if buffer_end - buffer_start >= 30.0:
            resegmented.append({
                "start": buffer_start,
                "end": buffer_end,
                "text": buffer_text.strip()
            })
            buffer_text = ""
            buffer_start = None
            buffer_end = None

    if buffer_text:
        resegmented.append({
            "start": buffer_start,
            "end": buffer_end,
            "text": buffer_text.strip()
        })

    # --- Sauvegarde SRT ---
    srt_path = Path(DOSSIER_SRT) / f"{nom}.srt"
    with open(srt_path, "w", encoding="utf-8") as f_srt:
        for i, seg in enumerate(resegmented, 1):
            start_srt = seconds_to_srt_time(seg["start"])
            end_srt = seconds_to_srt_time(seg["end"])
            f_srt.write(f"{i}\n{start_srt} --> {end_srt}\n{seg['text']}\n\n")
    print(f"✅ SRT sauvegardé : {srt_path}")

    # --- Sauvegarde Resume TXT avec ponctuation respectée ---
    resume_path = Path(DOSSIER_RESUME) / f"{nom}.txt"
    with open(resume_path, "w", encoding="utf-8") as f_resume:
        full_text = " ".join(seg["text"] for seg in resegmented)
        f_resume.write(full_text.strip())
    print(f"✅ Resume sauvegardé : {resume_path}")

    # --- Sauvegarde CSV des blocs ---
    blocs_path = Path(DOSSIER_BLOCS) / f"{nom}.csv"
    df = pd.DataFrame(resegmented)
    df.to_csv(blocs_path, index=False, encoding="utf-8")
    print(f"✅ Blocs CSV sauvegardé : {blocs_path}")

print("\n🎉 Tous les fichiers ont été traités avec succès !")
