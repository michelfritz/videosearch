import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

# Chargement du modèle de vectorisation
print("🔍 Chargement du modèle...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chargement du fichier CSV
print("📄 Chargement du fichier CSV...")
try:
    df = pd.read_csv("blocs_de_transcription.csv")
except FileNotFoundError:
    print("❌ Erreur : fichier blocs_de_transcription.csv introuvable.")
    exit()

# Vérification des colonnes attendues
required_columns = {"start", "end", "text"}
if not required_columns.issubset(df.columns):
    print(f"❌ Erreur : le fichier CSV doit contenir les colonnes : {required_columns}")
    print(f"Colonnes trouvées : {df.columns.tolist()}")
    exit()

# Nettoyage des textes
df["text"] = df["text"].fillna("").astype(str)

# Vectorisation
print("🧠 Vectorisation des blocs...")
vectors = model.encode(df["text"].tolist(), show_progress_bar=True)

# Sauvegarde dans un fichier .pkl avec métadonnées (start, end, speaker, text)
print("💾 Sauvegarde des vecteurs...")
with open("vecteurs.pkl", "wb") as f:
    pickle.dump({
        "vectors": vectors,
        "metadata": df[["start", "end", "speaker", "text"]].to_dict(orient="records")
    }, f)

print("✅ Fichier vecteurs.pkl généré avec succès !")
