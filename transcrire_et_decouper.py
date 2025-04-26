import os
import pandas as pd
import whisper
from pathlib import Path
import json

# --- Paramètres ---
DOSSIER_VIDEOS = "videos"
DOSSIER_JSON = "json"
DOSSIER_SRT = "srt"
DOSSIER_RESUME = "resume"
DOSSIER_BLOCS = "blocs"
GLOSSAIRE_PATH = "glossaire.csv"

# --- Chargement du glossaire si disponible ---
if os.path.exists(GLOSSAIRE_PATH):
    try:
        try:
            df_glossaire = pd.read_csv(GLOSSAIRE_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df_glossaire = pd.read_csv(GLOSSAIRE_PATH, encoding="cp1252")
        termes_glossaire = df_glossaire["mot"].dropna().tolist()
        print(f"📚 Glossaire chargé avec {len(termes_glossaire)} mots.")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du glossaire : {e}")
        termes_glossaire = []
else:
    termes_glossaire = []
    print("ℹ️ Aucun glossaire trouvé.")

# --- Chargement du modèle Whisper ---
print("🔁 Chargement du modèle Whisper...")
model = whisper.load_model("medium")
print(f"✅ Modèle chargé sur : {'cuda' if whisper.torch.cuda.is_available() else 'cpu'}")

# --- Création des dossiers de sortie ---
os.makedirs(DOSSIER_JSON, exist_ok=True)
os.makedirs(DOSSIER_SRT, exist_ok=True)
os.makedirs(DOSSIER_RESUME, exist_ok=True)
os.makedirs(DOSSIER_BLOCS, exist_ok=True)

# --- Liste des vidéos à traiter ---
videos = sorted(Path(DOSSIER_VIDEOS).glob("*.mp4"))
print(f"🎮 Vidéos trouvées : {[v.name for v in videos]}")

# --- Fonction pour convertir secondes en format SRT ---
def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# --- Traitement vidéo par vidéo ---
for chemin_video in videos:
    nom_video = chemin_video.stem
    json_path = Path(DOSSIER_JSON) / f"{nom_video}.json"

    if json_path.exists():
        print(f"📝 JSON déjà existant pour {nom_video}, saut transcription.")
    else:
        print(f"🔊 Transcription : {nom_video}")
        result = model.transcribe(
            str(chemin_video),
            language="fr",
            verbose=True,
            fp16=(whisper.torch.cuda.is_available()),
            initial_prompt=" ".join(termes_glossaire) if termes_glossaire else None
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON sauvegardé : {json_path}")

    # --- Lecture du JSON pour resegmenter ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    # --- Sauvegarde SRT à partir des segments originaux ---
    srt_path = Path(DOSSIER_SRT) / f"{nom_video}.srt"
    with open(srt_path, "w", encoding="utf-8") as f_srt:
        for i, seg in enumerate(segments, 1):
            start_srt = seconds_to_srt_time(seg["start"])
            end_srt = seconds_to_srt_time(seg["end"])
            f_srt.write(f"{i}\n{start_srt} --> {end_srt}\n{seg['text']}\n\n")
    print(f"✅ SRT sauvegardé : {srt_path}")

    # --- Résegmenter pour les blocs fixes de 30s ---
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

    # --- Sauvegarde Resume TXT ---
    resume_path = Path(DOSSIER_RESUME) / f"{nom_video}.txt"
    with open(resume_path, "w", encoding="utf-8") as f_resume:
        full_text = " ".join(seg["text"] for seg in resegmented)
        f_resume.write(full_text.strip())
    print(f"✅ Resume sauvegardé : {resume_path}")

    # --- Sauvegarde CSV des blocs ---
    blocs_path = Path(DOSSIER_BLOCS) / f"{nom_video}.csv"
    df = pd.DataFrame(resegmented)
    df.to_csv(blocs_path, index=False, encoding="utf-8")
    print(f"✅ Blocs CSV sauvegardé : {blocs_path}")

print("\n🎉 Toutes les vidéos ont été traitées avec succès !")