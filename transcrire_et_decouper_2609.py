import os
import json
import shutil
import subprocess
import shlex
import tempfile
from pathlib import Path

import pandas as pd
import whisper

# --- Paramètres ---
DOSSIER_VIDEOS = "videos"
DOSSIER_JSON = "json"
DOSSIER_SRT = "srt"
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
DOSSIER_BLOCS = "blocs"
GLOSSAIRE_PATH = "glossaire.csv"

# Options de décodage robustes (limite les hallucinations / répétitions)
DECODE_OPTS = dict(
    language="fr",                 # mets None pour auto-détection si besoin
    task="transcribe",
    temperature=0.0,               # pas de sampling
    beam_size=5,                   # beam search
    best_of=None,                  # ignoré en beam search
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    logprob_threshold=-0.5,
    no_speech_threshold=0.6,
)

# --- Utilitaires ---
def ensure_ffmpeg_in_path():
    """Essaie d'ajouter automatiquement des chemins FFmpeg connus au PATH.
    Lève une erreur claire si ffmpeg est introuvable."""
    candidats = [
        r"C:\\ffmpeg\\bin",  # install manuelle
        os.path.expanduser(r"~\\scoop\\apps\\ffmpeg\\current\\bin"),  # Scoop
        r"C:\\ProgramData\\chocolatey\\bin",  # Chocolatey
    ]
    for p in candidats:
        if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg introuvable. Installe-le (Scoop/Choco) ou ajoute C\\ffmpeg\\bin au PATH."
        )


def extract_wav_16k_mono(path_in: str) -> str:
    """Extrait une piste WAV mono 16 kHz propre via ffmpeg et renvoie le chemin temporaire."""
    ensure_ffmpeg_in_path()
    tmp_wav = os.path.join(tempfile.gettempdir(), Path(path_in).stem + "_16k.wav")
    cmd = f'ffmpeg -y -i "{path_in}" -vn -ac 1 -ar 16000 -f wav "{tmp_wav}"'
    # Capture la sortie pour éviter le spam console; remonte une erreur propre si échec
    cp = subprocess.run(shlex.split(cmd), capture_output=True)
    if cp.returncode != 0 or not os.path.exists(tmp_wav):
        raise RuntimeError(f"ffmpeg a échoué: {cp.stderr.decode(errors='ignore')[:500]}")
    return tmp_wav


def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# --- Chargement du glossaire si disponible ---
if os.path.exists(GLOSSAIRE_PATH):
    try:
        try:
            df_glossaire = pd.read_csv(GLOSSAIRE_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df_glossaire = pd.read_csv(GLOSSAIRE_PATH, encoding="cp1252")
        termes_glossaire = df_glossaire["mot"].dropna().astype(str).tolist()
        print(f"📚 Glossaire chargé avec {len(termes_glossaire)} mots.")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du glossaire : {e}")
        termes_glossaire = []
else:
    termes_glossaire = []
    print("ℹ️ Aucun glossaire trouvé.")

# Prompt initial (évite d'envoyer 10k tokens)
if termes_glossaire:
    # limite raisonnable du prompt (ex: 200 premiers termes)
    prompt_glossaire = " ".join(termes_glossaire[:200])
else:
    prompt_glossaire = None

# --- Chargement du modèle Whisper avec fallback GPU→CPU ---
print("🔁 Chargement du modèle Whisper...")
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = whisper.load_model("large-v3", device=device)
    print(f"✅ Modèle chargé sur : {device}")
except RuntimeError as e:
    # Fallback si build CUDA incompatible (ex: no kernel image...)
    if "no kernel image is available" in str(e) or "CUDA error" in str(e):
        print("⚠️ CUDA indisponible pour ce build de PyTorch → bascule sur CPU.")
        device = "cpu"
        model = whisper.load_model("large-v3", device=device)
        print("✅ Modèle rechargé sur : cpu")
    else:
        raise

# --- Création des dossiers de sortie ---
os.makedirs(DOSSIER_JSON, exist_ok=True)
os.makedirs(DOSSIER_SRT, exist_ok=True)
os.makedirs(DOSSIER_RESUME, exist_ok=True)
os.makedirs(DOSSIER_BLOCS, exist_ok=True)

# --- Liste des vidéos à traiter ---
videos = sorted(Path(DOSSIER_VIDEOS).glob("*.mp4"))
print(f"🎮 Vidéos trouvées : {[v.name for v in videos]}")

# --- Traitement vidéo par vidéo ---
for chemin_video in videos:
    nom_video = chemin_video.stem
    json_path = Path(DOSSIER_JSON) / f"{nom_video}.json"

    if json_path.exists():
        print(f"📝 JSON déjà existant pour {nom_video}, saut transcription.")
    else:
        print(f"🔊 Transcription : {nom_video}")

        # Pré-extraction audio en WAV 16 kHz (plus robuste)
        try:
            audio_input = extract_wav_16k_mono(str(chemin_video))
        except Exception as e:
            print(f"⚠️ Extraction WAV échouée ({e}), on tente directement la vidéo…")
            audio_input = str(chemin_video)

        # Transcription avec options robustes
        result = model.transcribe(
            audio_input,
            verbose=True,
            fp16=(device == "cuda"),
            initial_prompt=prompt_glossaire,
            **DECODE_OPTS,
        )

        # Sauvegarde JSON complet
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
            text = seg.get("text", "").strip()
            if not text:
                continue
            f_srt.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")
    print(f"✅ SRT sauvegardé : {srt_path}")

    # --- Résegmenter pour les blocs fixes de 30s ---
    resegmented = []
    buffer_text = ""
    buffer_start = None
    buffer_end = None

    for seg in segments:
        t = seg.get("text", "").strip()
        if not t:
            continue
        if buffer_start is None:
            buffer_start = seg["start"]
        buffer_text += (" " if buffer_text else "") + t
        buffer_end = seg["end"]
        if buffer_end - buffer_start >= 30.0:
            resegmented.append({
                "start": buffer_start,
                "end": buffer_end,
                "text": buffer_text.strip(),
            })
            buffer_text = ""
            buffer_start = None
            buffer_end = None

    if buffer_text:
        resegmented.append({
            "start": buffer_start,
            "end": buffer_end,
            "text": buffer_text.strip(),
        })

    # --- Sauvegarde Resume TXT ---
    resume_path = Path(DOSSIER_RESUME) / f"{nom_video}.txt"
    with open(resume_path, "w", encoding="utf-8") as f_resume:
        full_text = " ".join(seg["text"] for seg in resegmented if seg.get("text"))
        f_resume.write(full_text.strip())
    print(f"✅ Resume sauvegardé : {resume_path}")

    # --- Sauvegarde CSV des blocs ---
    blocs_path = Path(DOSSIER_BLOCS) / f"{nom_video}.csv"
    df = pd.DataFrame(resegmented)
    df.to_csv(blocs_path, index=False, encoding="utf-8")
    print(f"✅ Blocs CSV sauvegardé : {blocs_path}")

print("\n🎉 Toutes les vidéos ont été traitées avec succès !")
