#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import openai
import pandas as pd
from pathlib import Path
from incremental_utils import compute_fingerprint, should_skip, mark_done

# --- Paramètres ---
DOSSIER_RESUME = os.getenv("DOSSIER_RESUME", r"C:\Transcript\Dropbox (Personal)\resume")
FICHIER_SORTIE = os.getenv("FICHIER_SORTIE", "idees.csv")

# Contrôles d'incrémentalité
FORCE_REBUILD   = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

openai.api_key = os.getenv("OPENAI_API_KEY")

SCRIPT_NAME = "idees"

def extraire_idees(texte_resume: str):
    prompt = (
        "Voici le résumé d'une vidéo. Extrait entre 5 et 10 idées principales concrètes et actionnables, "
        "adaptées à l'immobilier ou au business. "
        "Formate la réponse sous forme d'une liste numérotée sans texte additionnel.\n\n"
        f"Texte :\n{texte_resume}\n"
    )
    resp = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    texte_brut = resp.choices[0].message.content or ""
    idees = []
    for ligne in texte_brut.splitlines():
        ln = ligne.strip()
        if not ln: continue
        ln = ln.split(".", 1)[-1].strip()
        if ln: idees.append(ln)
    return idees

def _load_existing_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["fichier", "idees"])
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["fichier", "idees"])
    for c in ("fichier","idees"):
        if c not in df.columns:
            df[c] = ""
    return df

def _upsert_row(df: pd.DataFrame, fichier: str, idees_join: str) -> pd.DataFrame:
    if "fichier" not in df.columns:
        df["fichier"] = ""
    mask = df["fichier"].astype(str) == str(fichier)
    if mask.any():
        df.loc[mask, "idees"] = idees_join
    else:
        df = pd.concat([df, pd.DataFrame([{"fichier": fichier, "idees": idees_join}])], ignore_index=True)
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

        # Mode UPDATE_EXISTING/ FORCE_REBUILD géré par should_skip
        if should_skip(SCRIPT_NAME, stem, fp):
            print(f"[SKIP] Déjà traité (sentinelle ok) -> {stem}")
            continue

        print(f"[RUN] Extraction d'idées -> {stem}")
        idees = extraire_idees(texte)
        join = " | ".join(idees)

        df = _upsert_row(existing, stem, join)
        df.drop_duplicates(subset=["fichier"], keep="last", inplace=True)
        df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")

        mark_done(SCRIPT_NAME, stem, str(fichier_txt), fp, FICHIER_SORTIE)

    print(f"[OK] Idées à jour -> {FICHIER_SORTIE}")

if __name__ == "__main__":
    main()
