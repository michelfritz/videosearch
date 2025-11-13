#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interrogation_inc.py — version incrémentale
- Charge `blocs_fusionnes.csv` et construit des Documents avec un identifiant stable: <fichier>__start_<start>.
- Si `faiss_transcripts/` existe, **ajoute uniquement** les nouveaux documents (ids inconnus).
- Sinon, construit l'index depuis zéro.
"""
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

FAISS_DIR = "faiss_transcripts"

def _doc_id(fichier: str, start) -> str:
    try:
        start = int(start)
    except Exception:
        try:
            start = int(float(start))
        except Exception:
            start = 0
    return f"{fichier}__start_{start}"

def main():
    # 1) Charger les blocs
    df = pd.read_csv("blocs_fusionnes.csv", encoding="utf-8")
    textes = df["text"].astype(str).tolist()
    urls   = df["url"].astype(str).tolist()
    fichiers = df["fichier"].astype(str).tolist()
    starts   = df["start"].tolist()

    docs = []
    for t, u, f, s in zip(textes, urls, fichiers, starts):
        docs.append(Document(page_content=t, metadata={"url": u, "id": _doc_id(f, s)}))

    # 2) Embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY manquant pour construire FAISS.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 3) Charger ou créer l'index
    vectordb = None
    if os.path.isdir(FAISS_DIR):
        try:
            vectordb = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            print("[OK] Index FAISS existant chargé.")
        except Exception as e:
            print(f"[WARN] Chargement FAISS impossible ({e}) -> reconstruction complète.")
            vectordb = None

    if vectordb is None:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(FAISS_DIR)
        print("✅ Base FAISS créée (full rebuild).")
        return

    # 4) Ajout incrémental
    existing_ids = set()
    try:
        # Accès direct au docstore interne (standard pour FAISS LangChain)
        existing_ids = {d.metadata.get("id") for d in vectordb.docstore._dict.values()}
    except Exception:
        existing_ids = set()

    new_docs = [d for d in docs if d.metadata.get("id") not in existing_ids]

    if not new_docs:
        print("✅ FAISS déjà à jour (aucun nouveau document).")
        return

    vectordb.add_documents(new_docs)
    vectordb.save_local(FAISS_DIR)
    print(f"✅ {len(new_docs)} nouveaux documents ajoutés à FAISS.")

if __name__ == "__main__":
    main()
