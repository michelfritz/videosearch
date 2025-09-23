
"""
resumer_court_video_v2.py
-------------------------
Version "safe" qui :
- Lit le CSV en priorité en UTF‑8 (avec/sans BOM), puis en dernier recours cp1252
- Supprime tout BOM résiduel dans les NOMS DE COLONNES et dans les cellules
- N'écrit JAMAIS d'autres colonnes que `resume`
- N'ajoute AUCUNE ligne : on remplit seulement les lignes existantes qui matchent `fichier`
- Permet de choisir l'encodage de sortie via --out_encoding (défaut: utf-8-sig pour Excel)

Usage (PowerShell) :
  $env:OPENAI_API_KEY="votre_cle_api"
  python resumer_court_video_v2.py --csv_path "C:\\Transcript\\urls.csv"
  python resumer_court_video_v2.py --dir "C:\\Transcript\\Dropbox (Personal)\\resume" --csv_path "C:\\Transcript\\urls.csv" --out_encoding "utf-8-sig"
  python resumer_court_video_v2.py --csv_path "C:\\Transcript\\urls.csv" --force
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

# OpenAI SDK v1+
try:
    from openai import OpenAI
except Exception:
    print("❌ Le paquet 'openai' n'est pas installé ou trop ancien. Faites : pip install --upgrade openai")
    raise

# --------- Constantes par défaut ---------
DEFAULT_DIR = r"C:\Transcript\Dropbox (Personal)\resume"
DEFAULT_CSV_PATH = "urls.csv"
SUMMARY_MODEL = "gpt-4o-mini"
MAX_CHARS_DIRECT = 12000   # au-delà, résumé hiérarchique

TEXT_EXTS = {".txt", ".md", ".srt", ".vtt"}
BOM = "\ufeff"

# ---------- Utilitaires CSV ----------
def read_csv_safely(csv_path: str) -> pd.DataFrame:
    """Lit le CSV en essayant utf-8-sig, utf-8, puis cp1252. Nettoie les BOM dans noms de colonnes et cellules."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Echec lecture CSV avec utf-8-sig/utf-8/cp1252: {last_err}")

    # 1) Nettoyer NOMS DE COLONNES (BOM éventuel)
    df.columns = [str(c).replace(BOM, "").strip() for c in df.columns]

    # 2) Nettoyer cellules texte (BOM résiduel en tête)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).map(lambda x: x.replace(BOM, "").strip())

    return df


def write_csv(df: pd.DataFrame, out_path: str, out_encoding: str = "utf-8-sig") -> None:
    """Ecrit le CSV avec l'encodage demandé (utf-8-sig par défaut pour Excel)."""
    df.to_csv(out_path, index=False, encoding=out_encoding)


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def normalize_key(s: str) -> str:
    """Normalise une clé de correspondance : trim, lower, suppression extension."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    base, _ext = os.path.splitext(s)
    return base or s


# ---------- Lecture/Nettoyage texte ----------
_timecode_srt = re.compile(r"^\d{1,3}:\d{2}:\d{2}[,\.]\d{2,3}\s*-->\s*\d{1,3}:\d{2}:\d{2}[,\.]\d{2,3}")
_timecode_vtt = re.compile(r"^\d{1,3}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{1,3}:\d{2}:\d{2}\.\d{3}")

def read_text_file(path: Path) -> str:
    """Lit un fichier texte/transcription et retire timecodes + numéros SRT/VTT."""
    ext = path.suffix.lower()
    content = path.read_text(encoding="utf-8", errors="ignore")

    if ext in {".srt", ".vtt"}:
        lines = []
        for line in content.splitlines():
            l = line.strip()
            # Ignore numéros de blocs SRT, marqueurs VTT et timecodes
            if not l or l.upper() == "WEBVTT":
                continue
            if l.isdigit():
                continue
            if _timecode_srt.match(l) or _timecode_vtt.match(l):
                continue
            lines.append(l)
        content = " ".join(lines)

    return content.strip()


# ---------- Résumé ----------
def chunk_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        cut = text.rfind(" ", start, end)
        if cut == -1 or cut < start + int(0.5 * max_chars):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return chunks


def summarize_text(client: "OpenAI", text: str) -> str:
    """Résumé 1–2 phrases, ≤ ~40 mots, style informatif neutre."""
    system_msg = (
        "Tu écris des résumés ultra courts en français, 1–2 phrases, ≤ 40 mots. "
        "Style informatif, neutre et clair. Pas de puces, pas d'intro, pas de titres, pas d'emoji."
    )
    user_msg = (
        "Voici le contenu d’une vidéo (texte déjà transcrit). "
        "Écris UN résumé TRÈS court (1–2 phrases, ≤ 40 mots) qui synthétise toute la vidéo."
        "\n\nTEXTE:\n"
        f"{text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


def smart_summarize(client: "OpenAI", transcript: str) -> str:
    if len(transcript) <= MAX_CHARS_DIRECT:
        return summarize_text(client, transcript)
    parts = chunk_text(transcript, MAX_CHARS_DIRECT)
    partials = [summarize_text(client, p) for p in parts]
    return summarize_text(client, " ".join(partials))


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Résumer des fichiers texte et écrire dans la colonne 'resume' d’un CSV.")
    parser.add_argument("--dir", type=str, default=DEFAULT_DIR, help="Dossier contenant les fichiers texte.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Chemin du CSV cible (par défaut: urls.csv).")
    parser.add_argument("--force", action="store_true", help="Recalculer même si un résumé existe déjà.")
    parser.add_argument("--out_encoding", type=str, default="utf-8-sig", help="Encodage de sortie (utf-8-sig par défaut).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY non défini. Définissez-le puis relancez.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"❌ Dossier introuvable: {base_dir}")
        sys.exit(1)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ CSV introuvable: {csv_path}")
        sys.exit(1)

    # 1) Lecture CSV sûre + colonnes
    df = read_csv_safely(str(csv_path))
    df = ensure_columns(df, ["fichier", "resume"])

    # 2) Index sur 'fichier' normalisé
    key_to_row: Dict[str, int] = {}
    duplicates = set()
    for i, v in enumerate(df["fichier"]):
        k = normalize_key(v)
        if not k:
            continue
        if k in key_to_row:
            duplicates.add(k)
        else:
            key_to_row[k] = i

    if duplicates:
        print(f"⚠️ Doublons détectés dans 'fichier' pour {len(duplicates)} clé(s). Le premier match sera utilisé.")

    # 3) Parcours des fichiers source
    files = [p for p in base_dir.iterdir() if p.suffix.lower() in TEXT_EXTS and p.is_file()]
    if not files:
        print(f"⚠️ Aucun fichier texte .txt/.md/.srt/.vtt trouvé dans: {base_dir}")
        sys.exit(0)

    updated = 0
    skipped_no_match = 0
    skipped_already = 0

    for path in tqdm(files, desc="📝 Résumés"):
        base = path.stem  # nom sans extension
        raw_text = read_text_file(path)
        if not raw_text:
            continue

        k = normalize_key(base)
        row_idx = key_to_row.get(k)
        if row_idx is None:
            print(f"⚠️ Aucun match CSV pour le transcript '{path.name}' (clé='{k}'). Transcript ignoré.")
            skipped_no_match += 1
            continue

        # Si résumé déjà présent et pas --force, on saute
        existing = str(df.loc[row_idx, "resume"]) if "resume" in df.columns else ""
        if not args.force and existing and str(existing).strip():
            skipped_already += 1
            continue

        try:
            summary = smart_summarize(client, raw_text)
            summary = " ".join(summary.split())
        except Exception as e:
            print(f"❌ Erreur résumé {path.name}: {e}")
            continue

        df.loc[row_idx, "resume"] = summary
        updated += 1

    # 4) Sauvegardes
    bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    write_csv(df.copy(), str(bak_path), args.out_encoding)
    write_csv(df, str(csv_path), args.out_encoding)

    print(f"✅ Fini. Résumés écrits pour {updated} fichier(s).")
    print(f"⏭️ Déjà remplis (ignorés): {skipped_already}")
    print(f"❓ Sans correspondance CSV (ignorés): {skipped_no_match}")
    print(f"💾 Sauvegarde: {bak_path}")
    print(f"📝 CSV mis à jour: {csv_path}")


if __name__ == "__main__":
    main()
