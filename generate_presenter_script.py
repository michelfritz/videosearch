# generate_presenter_scripts_per_file.py
# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime
from openai import OpenAI

# === Répertoire contenant les résumés ===
INPUT_DIR = r"C:\Transcript\Dropbox (Personal)\resume"

# === Dossier de sortie "presentation" situé au même niveau que "resume" ===
PARENT = os.path.dirname(INPUT_DIR)
OUTPUT_DIR = os.path.join(PARENT, "presentation")

MODEL = "gpt-4o-mini"   # Essayez "gpt-4o" si besoin de rendu encore plus léché

def read_text(path: str) -> str:
    """Lit un fichier texte en UTF-8, fallback cp1252 si besoin."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", errors="replace") as f:
            return f.read().strip()

def build_prompt_single(file_name: str, content: str) -> str:
    """Construit l’invite pour UNE vidéo."""
    guidance = (
        "Tu es auteur·rice de voix-off. À partir du résumé ci-dessous, "
        "écris un script prêt à être lu à voix haute par un·e présentateur·rice, "
        "durée cible ~4 à 5 minutes (≈ 650 à 750 mots).\n\n"
        "Exigences :\n"
        "- Français naturel, fluide, sans jargon inutile.\n"
        "- Structure claire : accroche, contexte, 3–6 points clés avec transitions, "
        "micro-récap, conclusion (appel à l’action si pertinent).\n"
        "- Ton: accessible, sûr, dynamique mais posé.\n"
        "- Évite les listes sèches ; privilégie la narration.\n"
        "- Aucune indication caméra/scène ; seulement le texte à dire.\n"
        "- Reste fidèle au résumé, pas d’invention de faits.\n"
        "- Orthographe/ponctuation soignées, phrases respirables pour l’oral.\n"
    )
    block = f"### Résumé : {file_name}\n{content}"
    return guidance + "\n\n===== RÉSUMÉ =====\n" + block

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_for_file(client: OpenAI, in_path: str) -> str:
    """Génère le script pour un fichier de résumé donné. Retourne le chemin de sortie."""
    base = os.path.splitext(os.path.basename(in_path))[0]
    # Nom de sortie identifiable par la vidéo d’origine :
    out_path = os.path.join(OUTPUT_DIR, f"{base}__presentateur.txt")
    if os.path.exists(out_path):
        print(f"⏭️  Déjà traité, on saute : {os.path.basename(in_path)}")
        return out_path

    content = read_text(in_path)
    if not content:
        print(f"⚠️  Fichier vide, ignoré : {os.path.basename(in_path)}")
        return ""

    prompt = build_prompt_single(os.path.basename(in_path), content)

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "Tu es un·e script doctor qui écrit des voix-off nettes et oralisables."},
            {"role": "user", "content": prompt},
        ],
    )

    # Récupération du texte retourné
    try:
        output_text = resp.output_text
    except Exception:
        if hasattr(resp, "output") and len(resp.output) > 0 and hasattr(resp.output[0], "content"):
            parts = []
            for p in resp.output[0].content:
                if hasattr(p, "text") and hasattr(p.text, "value"):
                    parts.append(p.text.value)
            output_text = "\n".join(parts).strip()
        else:
            raise RuntimeError("Réponse inattendue de l’API OpenAI.")

    # Sauvegarde .txt (UTF-8, LF)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(output_text)

    return out_path

def main():
    ensure_dirs()
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not paths:
        raise SystemExit(f"Aucun .txt trouvé dans : {INPUT_DIR}")

    client = OpenAI()  # lit OPENAI_API_KEY depuis l’environnement

    done = 0
    for p in paths:
        print(f"🧾 Traitement : {os.path.basename(p)} …")
        out = generate_for_file(client, p)
        if out:
            print(f"✅ OK → {out}")
            done += 1

    print(f"\n🎉 Terminé. Scripts générés/présents : {done}")
    print(f"Dossier de sortie : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
