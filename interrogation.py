import os
import pickle
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Charger .env pour clé OpenAI
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Charger les vecteurs
with open("vecteurs.pkl", "rb") as f:
    vecteurs = pickle.load(f)

# Charger les textes
df_blocs = pd.read_csv("blocs_fusionnes.csv")
textes = df_blocs["text"].tolist()
urls = df_blocs["url"].tolist()

# Vérification rapide
assert len(vecteurs) == len(textes), "Mismatch entre vecteurs et textes"

# Créer les documents (avec url dans metadata)
documents = []
for texte, url in zip(textes, urls):
    documents.append(Document(page_content=texte, metadata={"url": url}))

# Embeddings OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Créer FAISS correctement
vectordb = FAISS.from_documents(documents, embeddings)

# Sauvegarder la base FAISS
vectordb.save_local("faiss_transcripts")

print("✅ Base FAISS créée et sauvegardée dans 'faiss_transcripts'")
