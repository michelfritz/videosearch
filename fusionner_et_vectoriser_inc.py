#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusionner_et_vectoriser_inc.py — version incrémentale + clé stricte + dé-dupe

- Conserve l'interface/usage original (mêmes arguments).
- Produit toujours `blocs_fusionnes.csv`.
- Utilise une clé STRICTE pour identifier les vidéos (accents/tirets/espaces unifiés),
  partagée avec urls.csv, afin d'éviter les doublons de blocs sur la même vidéo.
- Associe à chaque bloc l'URL correcte (si présente dans urls.csv) et un nom de fichier
  canonique (celui de urls.csv).
- Dé-duplique les blocs sur (clé_stricte, start, end, text) en privilégiant ceux qui ont
  une URL non vide.
- Calcule les embeddings **uniquement** pour les blocs nouveaux/modifiés,
  grâce à des sentinelles + un cache local `embeddings_cache.pkl`.
- Écrit `vecteurs.pkl` **aligné** avec l'ordre de `blocs_fusionnes.csv`.

Variables d'environnement utiles (cf. incremental_utils.py) :
- FORCE_REBUILD=1       : tout recalculer
- UPDATE_EXISTING=1     : ne recalculer que ce qui existe déjà et a changé
- TRAITEMENTS_DIR=...   : dossier des sentinelles/registre (défaut: traitements/)
"""

import os
import glob
import pickle
import argparse
import unicodedata
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from incremental_utils import compute_fingerprint, should_skip, mark_done

# ---------- Utils encodage/CSV ----------
BOM = "\ufeff"

def read_csv_safely(path: str) -> pd.DataFrame:
    """
    Lit un CSV en utf-8-sig -> utf-8 -> cp1252.
    Enlève le BOM des noms/valeurs.
    """
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Impossible de lire {path}. Dernière erreur: {last_err}")

    # Nettoyage noms de colonnes
    df.columns = [str(c).replace(BOM, "").strip() for c in df.columns]
    # Nettoyage BOM dans les cellules texte
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).map(lambda x: x.replace(BOM, "").strip())
    return df

# ---------- Normalisation stricte des clés vidéo ----------
def normalize_key_strict(s: str) -> str:
    """
    Normalise une clef de vidéo:
      - string, trim, lower
      - accents supprimés (NFKD)
      - espaces exotiques -> espace simple
      - tirets unicode unifiés en '-'
      - caractères non alphanum/._ remplacés par espace
      - espaces multiples condensés
      - suppression éventuelle de l'extension (.mp4 / .mkv / etc.)
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u2000-\u200B\u202F\u205F\u3000]", " ", s)
    s = re.sub(r"[\-‐-‒–—−]+", "-", s)
    s = re.sub(r"[^a-z0-9\-\._ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "." in s:
        s = s.rsplit(".", 1)[0]
    return s

# ---------- URLs & noms canoniques ----------
def build_urls_maps(urls_csv_path: str):
    """
    Construit:
      - url_map      : key_strict -> url
      - canonical_map: key_strict -> fichier (nom canonique tel que présent dans urls.csv)
    En cas de doublons pour une même key_strict, on garde en priorité
    la ligne qui a une URL non vide.
    """
    urls_df = read_csv_safely(urls_csv_path)

    # Colonne 'fichier'
    if "fichier" not in urls_df.columns:
        candidates = [
            c for c in urls_df.columns
            if c.lower().replace(BOM, "").strip() in (
                "fichier", "file", "filename", "nom_fichier",
                "fichier_video", "fichier.1"
            )
        ]
        if candidates:
            urls_df = urls_df.rename(columns={candidates[0]: "fichier"})
        else:
            raise KeyError(
                f"Colonne 'fichier' introuvable dans {urls_csv_path}. "
                f"Colonnes: {list(urls_df.columns)}"
            )

    if "url" not in urls_df.columns:
        raise KeyError(
            f"Colonne 'url' introuvable dans {urls_csv_path}. "
            f"Colonnes: {list(urls_df.columns)}"
        )

    urls_df = urls_df.dropna(subset=["fichier"]).copy()
    urls_df["url"] = urls_df["url"].fillna("").astype(str).str.strip()
    urls_df["key_strict"] = urls_df["fichier"].map(normalize_key_strict)

    # Priorité aux lignes avec URL
    urls_df["_prio"] = (urls_df["url"] != "").astype(int)
    urls_df = urls_df.sort_values(["key_strict", "_prio"], ascending=[True, False])

    # 1 ligne par clé stricte
    urls_df = urls_df.drop_duplicates(subset=["key_strict"], keep="first")

    url_map       = dict(zip(urls_df["key_strict"], urls_df["url"]))
    canonical_map = dict(zip(urls_df["key_strict"], urls_df["fichier"]))
    return url_map, canonical_map

# ---------- Fusion des blocs ----------
def load_blocs(blocs_glob: str = "blocs/*.csv") -> pd.DataFrame:
    """
    Charge tous les CSV de 'blocs/' et ajoute une colonne 'fichier' basée sur le nom de fichier.
    """
    files = glob.glob(blocs_glob)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {blocs_glob}")
    dfs = []
    for f in files:
        df = read_csv_safely(f)
        for col in ("start", "end", "text"):
            if col not in df.columns:
                raise KeyError(
                    f"Colonne manquante '{col}' dans {f}. "
                    f"Colonnes: {list(df.columns)}"
                )
        video_name = Path(f).stem.replace("_blocs", "")
        df["fichier"] = video_name
        dfs.append(df[["start", "end", "text", "fichier"]])
    return pd.concat(dfs, ignore_index=True)

# ---------- Cache embeddings ----------
CACHE_PATH = Path("embeddings_cache.pkl")
CACHE_MODEL_KEY = "text-embedding-3-small"  # Doit rester stable pour réutiliser le cache

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return pickle.load(open(CACHE_PATH, "rb"))
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

# ---------- Vectorisation incrémentale ----------
def _embed_texts(model: str, texts: list, batch_size: int) -> list:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant.")
    client = OpenAI(api_key=api_key)

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings (incrémental)"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )
        vectors.extend([d.embedding for d in resp.data])
    return vectors

def _key_for_row(fichier: str, start) -> str:
    try:
        start = int(start)
    except Exception:
        try:
            start = int(float(start))
        except Exception:
            start = 0
    return f"{fichier}__start_{start}"

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="urls.csv",
                    help="Chemin vers urls.csv (défaut: urls.csv)")
    ap.add_argument("--blocs_glob", default="blocs/*.csv",
                    help="Glob des blocs (défaut: blocs/*.csv)")
    ap.add_argument("--out_csv", default="blocs_fusionnes.csv",
                    help="CSV fusionné en sortie (défaut: blocs_fusionnes.csv)")
    ap.add_argument("--out_vec", default="vecteurs.pkl",
                    help="Fichier de vecteurs (défaut: vecteurs.pkl)")
    ap.add_argument("--batch", type=int, default=1000,
                    help="Taille de batch embeddings (défaut: 1000)")
    ap.add_argument("--skip-embed", dest="skip_embed", action="store_true",
                    help="Ne pas calculer les embeddings (debug)")
    ap.add_argument("--model", default=CACHE_MODEL_KEY,
                    help="Modèle d'embeddings OpenAI (défaut: text-embedding-3-small)")
    args = ap.parse_args()

    # 1) Blocs depuis blocs/*.csv
    blocs = load_blocs(args.blocs_glob)
    # Clé stricte pour identifier la vidéo
    blocs["key_strict"] = blocs["fichier"].map(normalize_key_strict)

    # 2) URLs + noms canoniques depuis urls.csv
    url_map, canonical_map = build_urls_maps(args.urls)

    # Associer URL + nom canonique si dispo
    blocs["url"] = blocs["key_strict"].map(url_map).fillna("")
    blocs["fichier"] = blocs["key_strict"].map(canonical_map).fillna(blocs["fichier"])

    # 3) Dé-dupe : pour chaque (clé_stricte, start, end, text), on garde
    #    en priorité la ligne avec URL non vide.
    blocs["_prio"] = (blocs["url"] != "").astype(int)
    blocs = blocs.sort_values(
        ["key_strict", "start", "end", "_prio"],
        ascending=[True, True, True, False]
    )
    blocs = blocs.drop_duplicates(
        subset=["key_strict", "start", "end", "text"],
        keep="first"
    )

    # 4) Écrire CSV (colonnes historiques)
    cols = ["fichier", "start", "end", "text", "url"]
    blocs[cols].to_csv(args.out_csv, index=False, encoding="utf-8")
    print(
        f"[OK] {len(blocs)} blocs fusionnés (clé stricte, URL mappées, dé-dupe effectuée).\n"
        f"[CSV] Fichier écrit: {args.out_csv} (colonnes: {cols})"
    )

    # 5) Embeddings — Incrémental
    if args.skip_embed:
        print("[SKIP] Embeddings sautés (--skip-embed)." )
        return

    cache = _load_cache()
    to_embed_texts = []
    to_embed_keys  = []
    to_embed_fps   = []
    vectors_cache  = {}

    # Préparer la liste (ordre du CSV)
    ordered_keys  = []
    ordered_texts = []

    for _, row in blocs.iterrows():
        fichier = str(row["fichier"])
        start   = row["start"]
        text    = str(row["text"] or "")
        key     = _key_for_row(fichier, start)

        ordered_keys.append(key)
        ordered_texts.append(text)

        fp = compute_fingerprint(args.model, text)

        # Cache + sentinelles
        entry = cache.get(key, {})
        cached_fp  = entry.get("fp")
        cached_vec = entry.get("vec")

        if should_skip("embeddings", key, fp) and cached_vec is not None and cached_fp == fp:
            vectors_cache[key] = cached_vec
            continue

        # A (re)calculer
        to_embed_keys.append(key)
        to_embed_texts.append(text)
        to_embed_fps.append(fp)

    # Calcul des nouveaux embeddings
    if to_embed_texts:
        new_vecs = _embed_texts(args.model, to_embed_texts, batch_size=args.batch)
        for key, fp, vec in zip(to_embed_keys, to_embed_fps, new_vecs):
            cache[key] = {"fp": fp, "vec": vec, "model": args.model}
            vectors_cache[key] = vec
            mark_done(
                "embeddings",
                key,
                args.out_csv,
                fp,
                args.out_vec,
                extra={"model": args.model}
            )

        _save_cache(cache)
        print(f"[OK] {len(to_embed_texts)} embeddings (ré)calculés et mis en cache.")
    else:
        print("[OK] Aucun nouvel embedding à calculer (cache + sentinelles à jour)." )

    # Construire la liste alignée avec le CSV
    vecteurs = []
    missing  = []
    for key in ordered_keys:
        vec = vectors_cache.get(key) or (cache.get(key, {}) or {}).get("vec")
        if vec is None:
            missing.append(key)
        vecteurs.append(vec)

    # Sécurité : s'il manque encore des vecteurs, on les calcule à la volée
    if missing:
        print(f"[WARN] {len(missing)} vecteurs manquants -> recalcul à la volée.")
        texts_missing = [ordered_texts[ordered_keys.index(k)] for k in missing]
        miss_vecs = _embed_texts(args.model, texts_missing, batch_size=args.batch)
        for k, v in zip(missing, miss_vecs):
            idx = ordered_keys.index(k)
            vecteurs[idx] = v
            fp = compute_fingerprint(args.model, ordered_texts[idx])
            cache[k] = {"fp": fp, "vec": v, "model": args.model}
            mark_done("embeddings", k, args.out_csv, fp, args.out_vec, extra={"model": args.model})
        _save_cache(cache)

    with open(args.out_vec, "wb") as f:
        pickle.dump(vecteurs, f)

    print(f"[OK] Vectorisation incrémentale terminée -> {args.out_vec} (N={len(vecteurs)})")

if __name__ == "__main__":
    main()
