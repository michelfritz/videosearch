#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_compact_index.py
Construit un index FAISS compact (PCA + OPQ + IVFPQ) à partir de vecteurs.pkl
et blocs_fusionnes.csv, puis écrit dans faiss_compact/ :

  - faiss_compact/pca.tf
  - faiss_compact/opq.tf
  - faiss_compact/index_ivfpq.faiss

Hypothèses :
- vecteurs.pkl : liste ou array (N, d) de float (embeddings)
- blocs_fusionnes.csv : N lignes, même ordre que vecteurs.pkl
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

VECTEURS_PATH = Path("vecteurs.pkl")
BLOCS_PATH = Path("blocs_fusionnes.csv")
OUT_DIR = Path("faiss_compact")

# Hyperparamètres compactage
TARGET_DIM = 512  # dim après PCA (tu peux descendre à 384 si besoin)
M = 32            # sous-vecteurs PQ / OPQ
NLIST = 128       # nb de cellules IVF (nb de "clusters" de l'index)
NBITS = 6         # bits par composante PQ (64 centroïdes au lieu de 256 -> plus de warning)


def main():
    if not VECTEURS_PATH.exists():
        raise FileNotFoundError(f"{VECTEURS_PATH} introuvable")
    if not BLOCS_PATH.exists():
        raise FileNotFoundError(f"{BLOCS_PATH} introuvable")

    OUT_DIR.mkdir(exist_ok=True)

    # 1) Charger les vecteurs
    print(f"[LOAD] {VECTEURS_PATH}")
    with open(VECTEURS_PATH, "rb") as f:
        vecteurs = pickle.load(f)

    X = np.asarray(vecteurs, dtype="float32")
    if X.ndim != 2:
        raise ValueError(f"vecteurs.pkl doit être (N, d), trouvé shape={X.shape}")

    n, d = X.shape
    print(f"[INFO] {n} vecteurs de dimension {d}")

    # (Optionnel) vérif avec blocs_fusionnes.csv
    df = pd.read_csv(BLOCS_PATH, encoding="utf-8")
    if len(df) != n:
        print(f"[WARN] blocs_fusionnes.csv a {len(df)} lignes, vecteurs.pkl a {n} vecteurs.")

    # 2) PCA d -> TARGET_DIM
    print(f"[PCA] Entraînement PCA {d} -> {TARGET_DIM} ...")
    pca = faiss.PCAMatrix(d, TARGET_DIM)
    pca.train(X)
    Xr = pca.apply_py(X)  # (N, TARGET_DIM)

    # 3) OPQ + IVFPQ
    print("[OPQ+IVFPQ] Entraînement OPQ ...")
    opq = faiss.OPQMatrix(TARGET_DIM, M)
    opq.train(Xr)

    quantizer = faiss.IndexFlatL2(TARGET_DIM)
    index = faiss.IndexIVFPQ(quantizer, TARGET_DIM, NLIST, M, NBITS)

    print(f"[OPQ+IVFPQ] Entraînement IVF-PQ (NLIST={NLIST}, M={M}, NBITS={NBITS}) ...")
    n_train = min(20000, n)
    perm = np.random.permutation(n)[:n_train]
    train = Xr[perm]
    train_opq = opq.apply_py(train)
    index.train(train_opq)

    # 4) Ajout des vecteurs compressés
    print("[ADD] Ajout des vecteurs dans l'index compact ...")
    Xr_opq = opq.apply_py(Xr)
    index.add(Xr_opq)
    index.nprobe = 8

    # 5) Sauvegarde
    pca_path = OUT_DIR / "pca.tf"
    opq_path = OUT_DIR / "opq.tf"
    index_path = OUT_DIR / "index_ivfpq.faiss"

    faiss.write_VectorTransform(pca, str(pca_path))
    faiss.write_VectorTransform(opq, str(opq_path))
    faiss.write_index(index, str(index_path))

    print(f"[OK] PCA   -> {pca_path}")
    print(f"[OK] OPQ   -> {opq_path}")
    print(f"[OK] Index -> {index_path}")
    print("[DONE] Index compact prêt. Tu peux cesser de versionner vecteurs.pkl et l'ancien index FAISS poids lourd.")


if __name__ == "__main__":
    main()
