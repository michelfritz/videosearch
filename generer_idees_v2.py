#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generer_idees_v2.py — version corrigée + incrémentale

- Lit les blocs dans DOSSIER_BLOCS (CSV avec colonnes: text, start).
- Pour chaque bloc, génère 1 idée principale (business / immobilier).
- Fonctionne par lots de 10 blocs pour limiter les appels au modèle.
- Incrémentalité via incremental_utils (sentinelles + registre).
"""

import os
import re
from pathlib import Path

import openai
import pandas as pd

from incremental_utils import compute_fingerprint, should_skip, mark_done

# --- Paramètres ---
DOSSIER_BLOCS = os.getenv("DOSSIER_BLOCS", "blocs")
FICHIER_SORTIE = os.getenv("FICHIER_SORTIE", "idees_v2.csv")

# Contrôles standards (reconnus par incremental_utils)
FORCE_REBUILD = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

openai.api_key = os.getenv("OPENAI_API_KEY")

SCRIPT_NAME = "idees_v2"


# ---------- Utils ----------

def _coerce_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def _fallback_from_text(t: str, max_words: int = 12) -> str:
    """
    Plan B local si le modèle ne renvoie pas assez de lignes :
    on synthétise quelques mots à partir du texte du bloc.
    """
    s = re.sub(r"\s+", " ", (t or "").strip())
    # supprimer URLs et quelques caractères parasites
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[•·\u2022\-\(\)\[\]\{\}<>]", " ", s)
    words = [w for w in s.split(" ") if w]
    if not words:
        return "Idée clé du bloc"
    return " ".join(words[:max_words]).strip().rstrip(".")


def _normalize_list_lines(s: str) -> list[str]:
    """
    Transforme la réponse en liste de lignes dépouillées de la numérotation.
    Ex: "1. Faire X" -> "Faire X", "2) ..." -> "..."
    Supprime les lignes vides.
    """
    out: list[str] = []
    for raw in (s or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # enlever "1.", "2)", "- ", "* "
        ln = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*]\s+)", "", ln).strip()
        if ln:
            out.append(ln)
    return out


# --- Fonction pour extraire plusieurs idées en batch (N strict) ---

def extraire_idees_batch(textes_blocs: list[str]) -> list[str]:
    expected_n = len(textes_blocs)

    jointure = "\n\n".join(
        f"Bloc {i+1} : {texte}" for i, texte in enumerate(textes_blocs)
    )

    prompt = (
        f"Voici {expected_n} extraits d'une vidéo (en français). "
        f"POUR CHAQUE EXTRAIT, renvoie EXACTEMENT {expected_n} idées principales "
        f"(une par extrait), chacune en 1 phrase courte, concrète et actionnable, "
        f"adaptée au business et à l'immobilier. "
        f"Formate ta réponse en {expected_n} lignes NUMÉROTÉES 1..{expected_n}, "
        f"sans autre texte.\n\n"
        f"{jointure}\n"
    )

    try:
        reponse = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        texte_brut = reponse.choices[0].message.content or ""
    except Exception as e:
        print(f"[WARN] OpenAI chat error: {e}")
        texte_brut = ""

    idees = _normalize_list_lines(texte_brut)

    # Ajustement à N: PAD/TRONQUE pour assurer 1:1
    if len(idees) < expected_n:
        # compléter avec un fallback à partir du texte des blocs restants
        rest = [_fallback_from_text(t) for t in textes_blocs[len(idees):]]
        idees = idees + rest
    elif len(idees) > expected_n:
        idees = idees[:expected_n]

    # sécurité finale
    while len(idees) < expected_n:
        idees.append("Idée clé du bloc")

    return idees


def _load_existing_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["fichier", "idee", "start"])
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["fichier", "idee", "start"])

    # Colonnes minimales
    for col in ("fichier", "idee", "start"):
        if col not in df.columns:
            df[col] = "" if col != "start" else 0

    df["fichier"] = df["fichier"].astype(str)
    df["idee"] = df["idee"].astype(str)
    df["start"] = df["start"].apply(_coerce_int)

    return df[["fichier", "idee", "start"]]


# ---------- MAIN ----------

def main():
    existing = _load_existing_csv(FICHIER_SORTIE)
    fichiers_csv = list(Path(DOSSIER_BLOCS).glob("*.csv"))
    print(f"[INFO] Fichiers de blocs trouvés: {len(fichiers_csv)}")

    to_append: list[dict] = []

    for fichier_csv in fichiers_csv:
        nom_fichier = fichier_csv.stem
        print(f"\n✨ {SCRIPT_NAME}: {nom_fichier}")

        try:
            blocs_df = pd.read_csv(fichier_csv)
        except Exception as e:
            print(f"[WARN] Lecture impossible: {fichier_csv} ({e})")
            continue

        textes_batch: list[str] = []
        starts_batch: list[int] = []
        keys_batch: list[str] = []
        fps_batch: list[str] = []

        n_rows = len(blocs_df)

        for idx, row in blocs_df.iterrows():
            texte_bloc = str(row.get("text", "") or "").strip()
            start = _coerce_int(row.get("start", 0))

            key = f"{nom_fichier}__start_{start}"
            fp = compute_fingerprint(nom_fichier, start, texte_bloc)

            if (not FORCE_REBUILD) and should_skip(SCRIPT_NAME, key, fp):
                # déjà traité avec le même contenu
                continue

            if not texte_bloc:
                # même si vide -> on marque quand même "done" pour éviter les boucles infinies
                mark_done(
                    SCRIPT_NAME,
                    key,
                    str(fichier_csv),
                    fp,
                    FICHIER_SORTIE,
                    extra={"idee": ""},
                )
                continue

            textes_batch.append(texte_bloc)
            starts_batch.append(start)
            keys_batch.append(key)
            fps_batch.append(fp)

            is_last = (idx == n_rows - 1)

            # Paquet de 10 ou fin
            if len(textes_batch) == 10 or (is_last and textes_batch):
                idees = extraire_idees_batch(textes_batch)

                # 1 : 1 entre blocs et idées
                for idee, start_associe, key_i, fp_i in zip(
                    idees, starts_batch, keys_batch, fps_batch
                ):
                    to_append.append(
                        {
                            "fichier": nom_fichier,
                            "idee": idee,
                            "start": int(start_associe),
                        }
                    )
                    mark_done(
                        SCRIPT_NAME,
                        key_i,
                        str(fichier_csv),
                        fp_i,
                        FICHIER_SORTIE,
                        extra={"idee": idee},
                    )

                # reset batch
                textes_batch, starts_batch, keys_batch, fps_batch = [], [], [], []

    if not to_append:
        print("\n⚠️ Aucune nouveauté à écrire.")
        return

    df_new = pd.DataFrame(to_append, columns=["fichier", "idee", "start"])

    # Concat incrémentale (clé = (fichier, start))
    if FORCE_REBUILD or UPDATE_EXISTING:
        # on remplace les lignes existantes pour les mêmes (fichier, start)
        new_keys = set(
            (str(r["fichier"]), int(r["start"])) for _, r in df_new.iterrows()
        )
        mask_keep = ~existing.apply(
            lambda r: (str(r["fichier"]), int(r["start"])) in new_keys, axis=1
        )
        df_final = pd.concat([existing[mask_keep], df_new], ignore_index=True)
    else:
        # mode normal : on n’ajoute que ce qui n’existe pas déjà
        existing_keys = set(
            (str(r["fichier"]), int(r["start"])) for _, r in existing.iterrows()
        )
        mask_new = ~df_new.apply(
            lambda r: (str(r["fichier"]), int(r["start"])) in existing_keys, axis=1
        )
        df_final = pd.concat([existing, df_new[mask_new]], ignore_index=True)

    df_final.drop_duplicates(subset=["fichier", "start"], keep="last", inplace=True)
    df_final.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")

    print(
        f"\n📁 Idées v2 mises à jour dans {FICHIER_SORTIE} ✅ "
        f"({len(df_final)} lignes au total)"
    )


if __name__ == "__main__":
    main()
