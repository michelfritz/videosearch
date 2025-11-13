#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import openai
import pandas as pd
from pathlib import Path
from incremental_utils import compute_fingerprint, should_skip, mark_done

# --- Paramètres ---
DOSSIER_RESUME = os.getenv("DOSSIER_RESUME", r"C:\Transcript\Dropbox (Personal)\resume")
FICHIER_SORTIE = os.getenv("FICHIER_SORTIE", "themes.csv")

FORCE_REBUILD   = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

openai.api_key = os.getenv("OPENAI_API_KEY")
SCRIPT_NAME = "themes"

BAD_SEPARATORS = ["|", ";", ",", " / ", "/", "\\", "  "]

def sanitize_theme(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    for sep in BAD_SEPARATORS:
        if sep != "|":
            s = s.replace(sep, " ")
    s = " ".join(s.split())
    return s

def dedupe_preserve_order(items):
    seen = set(); out = []
    for x in items:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def extraire_themes(texte_resume):
    prompt = (
        "Voici le résumé d'une vidéo. Extrait entre 3 et 5 thèmes principaux très courts (2 à 4 mots max), "
        "mots clés synthétiques, pertinents pour l'immobilier ou le business. "
        "Réponds sous forme de liste numérotée, sans autre texte.\n\n"
        f"Texte :\n{texte_resume}\n"
    )
    resp = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    texte_brut = (resp.choices[0].message.content or "")
    themes = []
    for ligne in texte_brut.splitlines():
        ligne = ligne.strip()
        if not ligne: continue
        if "." in ligne[:3]:
            ligne = ligne.split(".", 1)[-1].strip()
        elif ")" in ligne[:3]:
            ligne = ligne.split(")", 1)[-1].strip()
        theme = sanitize_theme(ligne)
        if theme:
            themes.append(theme)
    themes = dedupe_preserve_order(themes)[:5]
    return themes

def _load_existing_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["fichier", "themes"])
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["fichier", "themes"])
    for c in ("fichier","themes"):
        if c not in df.columns:
            df[c] = ""
    return df

def _upsert_row(df: pd.DataFrame, fichier: str, themes_join: str) -> pd.DataFrame:
    mask = df["fichier"].astype(str) == str(fichier)
    if mask.any():
        df.loc[mask, "themes"] = themes_join
    else:
        df = pd.concat([df, pd.DataFrame([{"fichier": fichier, "themes": themes_join}])], ignore_index=True)
    return df

def main():
    existing = _load_existing_csv(FICHIER_SORTIE)
    resumes = list(Path(DOSSIER_RESUME).glob("*.txt"))
    print(f"[INFO] Résumés trouvés: {len(resumes)}")

    for fichier_txt in resumes:
        stem = fichier_txt.stem
        texte = fichier_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not texte:
            print(f"[SKIP] Résumé vide -> {stem}")
            continue

        fp = compute_fingerprint(stem, texte)

        if should_skip(SCRIPT_NAME, stem, fp):
            print(f"[SKIP] Déjà traité (sentinelle ok) -> {stem}")
            continue

        print(f"[RUN] Extraction de thèmes -> {stem}")
        themes = extraire_themes(texte)
        join = "|".join(themes)

        df = _upsert_row(existing, stem, join)
        df.drop_duplicates(subset=["fichier"], keep="last", inplace=True)
        df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8", sep=",", quoting=csv.QUOTE_MINIMAL)

        mark_done(SCRIPT_NAME, stem, str(fichier_txt), fp, FICHIER_SORTIE)

    print(f"[OK] Thèmes à jour -> {FICHIER_SORTIE}")

if __name__ == "__main__":
    main()
