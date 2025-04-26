import os
import re
import json
from pathlib import Path

# --- Dossiers ---
DOSSIER_SOURCE = "parser"
DOSSIER_JSON = "json"
DOSSIER_SRT = "srt"
DOSSIER_RESUME = "resume"

# --- Créer les dossiers si pas existants ---
os.makedirs(DOSSIER_JSON, exist_ok=True)
os.makedirs(DOSSIER_SRT, exist_ok=True)
os.makedirs(DOSSIER_RESUME, exist_ok=True)

# --- Fonction pour convertir timestamp format ---
def time_to_seconds(timestr):
    parts = timestr.split(":")
    if len(parts) == 2:  # format mm:ss.sss
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:  # format hh:mm:ss.sss
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Format de temps invalide: {timestr}")

    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total_seconds

def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# --- Traitement fichier par fichier ---
for fichier_txt in Path(DOSSIER_SOURCE).glob("*.txt"):
    nom = fichier_txt.stem
    print(f"\n✨ Traitement de {nom}...")

    segments = []

    with open(fichier_txt, "r", encoding="utf-8") as f:
        lignes = f.readlines()

    for ligne in lignes:
        match = re.match(r"\[(\d{1,2}:\d{2}(?::\d{2})?\.\d{3}) --> (\d{1,2}:\d{2}(?::\d{2})?\.\d{3})\]\s+(.*)", ligne)
        if match:
            start_str, end_str, text = match.groups()
            start_sec = time_to_seconds(start_str)
            end_sec = time_to_seconds(end_str)
            segments.append({
                "start": start_sec,
                "end": end_sec,
                "text": text.strip()
            })

    if not segments:
        print(f"⚠️ Aucun segment valide trouvé dans {nom}.txt. Fichier ignoré.")
        continue

    # --- Sauvegarde JSON ---
    json_path = Path(DOSSIER_JSON) / f"{nom}.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump({"language": "fr", "segments": segments}, f_json, ensure_ascii=False, indent=2)
    print(f"✅ JSON sauvegardé : {json_path}")

    # --- Sauvegarde SRT ---
    srt_path = Path(DOSSIER_SRT) / f"{nom}.srt"
    with open(srt_path, "w", encoding="utf-8") as f_srt:
        for i, seg in enumerate(segments, 1):
            start_srt = seconds_to_srt_time(seg["start"])
            end_srt = seconds_to_srt_time(seg["end"])
            f_srt.write(f"{i}\n{start_srt} --> {end_srt}\n{seg['text']}\n\n")
    print(f"✅ SRT sauvegardé : {srt_path}")

    # --- Sauvegarde Resume TXT ---
    resume_path = Path(DOSSIER_RESUME) / f"{nom}.txt"
    with open(resume_path, "w", encoding="utf-8") as f_resume:
        full_text = " ".join(seg["text"] for seg in segments)
        f_resume.write(full_text)
    print(f"✅ Resume sauvegardé : {resume_path}")

print("\n🎉 Tous les fichiers ont été traités avec succès !")