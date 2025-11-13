"""
fusionner_et_vectoriser.py (clean ASCII)
- Lecture robuste des CSV (BOM, encodage, colonnes manquantes).
- Construit 'blocs_fusionnes.csv' avec: fichier, start, end, text, url.
- Associe l'URL via urls.csv en appariant sur 'fichier' (normalise sans extension).
- Vectorisation OpenAI par batch (option --skip-embed).

Usage PowerShell:
  $env:OPENAI_API_KEY="...votre_cle..."
  python fusionner_et_vectoriser.py
  # ou sans vectorisation (debug):
  python fusionner_et_vectoriser.py --skip-embed
"""

import os
import glob
import pickle
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# ---------- Utils encodage/CSV ----------
BOM = "\ufeff"

def read_csv_safely(path: str) -> pd.DataFrame:
    """Lit un CSV en utf-8-sig -> utf-8 -> cp1252. Enleve le BOM des noms/valeurs."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Impossible de lire {path}. Derniere erreur: {last_err}")

    # Nettoyage noms de colonnes
    df.columns = [str(c).replace(BOM, "").strip() for c in df.columns]
    # Nettoyage BOM dans les cellules texte
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).map(lambda x: x.replace(BOM, "").strip())
    return df

def normalize_key(s: str) -> str:
    """Normalise une cle: string, trim, lower, sans extension."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    base = os.path.splitext(s)[0]
    return base or s

# ---------- Chargement URLs ----------
def build_urls_dict(urls_csv_path: str) -> Dict[str, str]:
    urls_df = read_csv_safely(urls_csv_path)
    # Gérer eventuels doublons/variantes
    if "fichier" not in urls_df.columns:
        candidates = [c for c in urls_df.columns if c.lower().replace(BOM, "").strip() in
                      ("fichier", "file", "filename", "nom_fichier", "fichier_video", "fichier.1")]
        if candidates:
            urls_df.rename(columns={candidates[0]: "fichier"}, inplace=True)
        else:
            raise KeyError(f"Colonne 'fichier' introuvable dans {urls_csv_path}. Colonnes: {list(urls_df.columns)}")
    if "url" not in urls_df.columns:
        raise KeyError(f"Colonne 'url' introuvable dans {urls_csv_path}. Colonnes: {list(urls_df.columns)}")

    urls_df = urls_df.dropna(subset=["fichier"])
    urls_df = urls_df.drop_duplicates(subset=["fichier"], keep="first")

    m = {}
    for _, row in urls_df.iterrows():
        key = normalize_key(row["fichier"])
        m[key] = str(row["url"]).strip()
    return m

# ---------- Fusion des blocs ----------
def load_blocs(blocs_glob: str = "blocs/*.csv") -> pd.DataFrame:
    files = glob.glob(blocs_glob)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouve dans {blocs_glob}")
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

# ---------- Vectorisation ----------
def vectorize_and_save(texts: List[str], batch_size: int = 1000,
                       model: str = "text-embedding-3-small",
                       out_path: str = "vecteurs.pkl"):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant. PowerShell: $env:OPENAI_API_KEY=\"...\"")
    client = OpenAI(api_key=api_key)

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Vectorisation"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        vectors.extend([d.embedding for d in resp.data])

    with open(out_path, "wb") as f:
        pickle.dump(vectors, f)
    return len(vectors)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="urls.csv", help="Chemin vers urls.csv (defaut: urls.csv)")
    ap.add_argument("--blocs_glob", default="blocs/*.csv", help="Glob des blocs (defaut: blocs/*.csv)")
    ap.add_argument("--out_csv", default="blocs_fusionnes.csv", help="CSV fusionne en sortie (defaut: blocs_fusionnes.csv)")
    ap.add_argument("--out_vec", default="vecteurs.pkl", help="Fichier de vecteurs (defaut: vecteurs.pkl)")
    ap.add_argument("--batch", type=int, default=1000, help="Taille de batch embeddings (defaut: 1000)")
    ap.add_argument("--skip-embed", dest="skip_embed", action="store_true", help="Ne pas calculer les embeddings (debug)")
    args = ap.parse_args()

    # 1) Blocs + URLs
    blocs = load_blocs(args.blocs_glob)
    urls_map = build_urls_dict(args.urls)

    # 2) Associer URL
    blocs["key"] = blocs["fichier"].map(normalize_key)
    blocs["url"] = blocs["key"].map(urls_map).fillna("")
    blocs.drop(columns=["key"], inplace=True)

    # 3) Ecrire CSV fusionne (inclut 'fichier' pour fallbacks Streamlit)
    cols = ["fichier", "start", "end", "text", "url"]
    blocs = blocs[cols]
    blocs.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"[OK] {len(blocs)} blocs fusionnes et enrichis (URL associees).")
    print(f"[CSV] Fichier ecrit: {args.out_csv} (colonnes: {cols})")

    # 4) Embeddings
    if not args.skip_embed:
        texts = blocs["text"].fillna("").astype(str).tolist()
        n = vectorize_and_save(texts, batch_size=args.batch, out_path=args.out_vec)
        print(f"[OK] Vectorisation terminee ({n} embeddings) -> {args.out_vec}")
    else:
        print("[SKIP] Embeddings sautes (--skip-embed).")

if __name__ == "__main__":
    main()
