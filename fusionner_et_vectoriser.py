#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusionner_et_vectoriser.py — FIX "clé stricte" + dé‑dupe
- Lit tous les blocs/CSV.
- Construit une clé STRICTE (NFKD, accents supprimés, espaces et tirets unifiés).
- Joint avec urls.csv sur cette clé stricte.
- Choisit un 'fichier' canonique depuis urls.csv quand possible.
- Dé‑duplique (clé_stricte, start, end, text) en préférant la ligne avec URL.
- Vectorisation inchangée.
"""

import os, glob, pickle, argparse, unicodedata, re
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

BOM = "\ufeff"

def read_csv_safely(path: str) -> pd.DataFrame:
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
    df.columns = [str(c).replace(BOM, "").strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).map(lambda x: x.replace(BOM, "").strip())
    return df

def normalize_key_strict(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # espaces insécables + espaces exotiques -> espace simple
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u2000-\u200B\u202F\u205F\u3000]", " ", s)
    # tous les tirets unicode -> '-'
    s = re.sub(r"[\-‐‑‒–—−]+", "-", s)
    # ne garder que basiques
    s = re.sub(r"[^a-z0-9\-\._ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # retirer extension éventuelle
    if "." in s:
        s = s.rsplit(".", 1)[0]
    return s

def build_urls_maps(urls_csv_path: str):
    df = read_csv_safely(urls_csv_path)
    if "fichier" not in df.columns:
        # essayer quelques alias
        for c in df.columns:
            if c.lower().replace(BOM, "").strip() in ("fichier","file","filename","nom_fichier","fichier_video","fichier.1"):
                df = df.rename(columns={c: "fichier"})
                break
        if "fichier" not in df.columns:
            raise KeyError(f"Colonne 'fichier' introuvable dans {urls_csv_path}. Colonnes: {list(df.columns)}")
    if "url" not in df.columns:
        raise KeyError(f"Colonne 'url' introuvable dans {urls_csv_path}. Colonnes: {list(df.columns)}")

    df = df.dropna(subset=["fichier"]).copy()
    df["key_strict"] = df["fichier"].map(normalize_key_strict)
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    # priorité aux lignes qui ont une URL
    df["_prio"] = (df["url"] != "").astype(int)
    df = df.sort_values(["key_strict","_prio"], ascending=[True, False])

    # 1 clé_strict -> 1 ligne (on garde celle avec URL si dispo)
    df = df.drop_duplicates(subset=["key_strict"], keep="first")

    url_map       = dict(zip(df["key_strict"], df["url"]))
    canonical_map = dict(zip(df["key_strict"], df["fichier"]))  # nom d'affichage “canonique”
    return url_map, canonical_map

def load_blocs(blocs_glob: str = "blocs/*.csv") -> pd.DataFrame:
    files = glob.glob(blocs_glob)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {blocs_glob}")
    dfs = []
    for f in files:
        df = read_csv_safely(f)
        for col in ("start", "end", "text"):
            if col not in df.columns:
                raise KeyError(f"Colonne manquante '{col}' dans {f}. Colonnes: {list(df.columns)}")
        video_name = Path(f).stem.replace("_blocs", "")
        df["fichier"] = video_name
        dfs.append(df[["start", "end", "text", "fichier"]])
    return pd.concat(dfs, ignore_index=True)

def vectorize_and_save(texts: List[str], batch_size: int = 1000,
                       model: str = "text-embedding-3-small",
                       out_path: str = "vecteurs.pkl"):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant.")
    client = OpenAI(api_key=api_key)
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Vectorisation"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        vectors.extend([d.embedding for d in resp.data])
    with open(out_path, "wb") as f:
        pickle.dump(vectors, f)
    return len(vectors)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="urls.csv")
    ap.add_argument("--blocs_glob", default="blocs/*.csv")
    ap.add_argument("--out_csv", default="blocs_fusionnes.csv")
    ap.add_argument("--out_vec", default="vecteurs.pkl")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--skip-embed", dest="skip_embed", action="store_true")
    args = ap.parse_args()

    # 1) Blocs
    blocs = load_blocs(args.blocs_glob)
    blocs["key_strict"] = blocs["fichier"].map(normalize_key_strict)

    # 2) URLs (clé stricte + nom canonique)
    url_map, canonical_map = build_urls_maps(args.urls)

    # 3) Associer URL + nom canonique si dispo
    blocs["url"] = blocs["key_strict"].map(url_map).fillna("")
    blocs["fichier"] = blocs["key_strict"].map(canonical_map).fillna(blocs["fichier"])

    # 4) Dé‑dupe : garder 1 ligne (clé_stricte, start, end, text), préférer celle avec URL
    blocs["_prio"] = (blocs["url"] != "").astype(int)
    blocs = blocs.sort_values(["key_strict","start","end","_prio"], ascending=[True, True, True, False])
    blocs = blocs.drop_duplicates(subset=["key_strict","start","end","text"], keep="first")

    # 5) Écrire CSV (colonnes historiques)
    cols = ["fichier", "start", "end", "text", "url"]
    blocs[cols].to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] {len(blocs)} blocs fusionnés (URL mappées, dé‑dupe effectuée).")
    print(f"[CSV] {args.out_csv}")

    # 6) Embeddings (inchangé)
    if not args.skip_embed:
        texts = blocs["text"].fillna("").astype(str).tolist()
        n = vectorize_and_save(texts, batch_size=args.batch, out_path=args.out_vec)
        print(f"[OK] Vectorisation terminée ({n} embeddings) -> {args.out_vec}")
    else:
        print("[SKIP] Embeddings sautés (--skip-embed).")

if __name__ == "__main__":
    main()
