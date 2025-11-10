import os
import pandas as pd
import openai
from pathlib import Path

# --- Paramètres ---
DOSSIER_BLOCS = "blocs"
FICHIER_SORTIE = "idees_v2.csv"

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Fonction pour extraire plusieurs idées en batch ---
def extraire_idees_batch(textes_blocs):
    jointure = "\n\n".join(f"Bloc {i+1} : {texte}" for i, texte in enumerate(textes_blocs))
    prompt = (
        "Voici plusieurs extraits d'une vidéo. Pour chaque extrait, extrait une idée principale concrète et actionnable, "
        "sous forme d'une phrase courte (ex: 'Prendre plus de mandats', 'Répondre aux objections'). "
        "Formate la réponse sous forme d'une liste numérotée, sans texte additionnel.\n\n"
        f"{jointure}\n"
    )

    reponse = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    texte_brut = reponse.choices[0].message.content
    idees = []
    for ligne in texte_brut.split("\n"):
        if ligne.strip():
            partie = ligne.split(".", 1)[-1].strip()
            if partie:
                idees.append(partie)
    return idees

# --- Traitement ---
resultats = []

for fichier_csv in Path(DOSSIER_BLOCS).glob("*.csv"):
    nom_fichier = fichier_csv.stem
    print(f"\n✨ Traitement de : {nom_fichier}")

    blocs_df = pd.read_csv(fichier_csv)

    textes_batch = []
    starts_batch = []

    for idx, row in blocs_df.iterrows():
        texte_bloc = row.get("text", "").strip()
        start = int(row.get("start", 0))

        if texte_bloc:
            textes_batch.append(texte_bloc)
            starts_batch.append(start)

        if len(textes_batch) == 10 or (idx == len(blocs_df) - 1 and textes_batch):
            idees_batch = extraire_idees_batch(textes_batch)
            for idee, start_associe in zip(idees_batch, starts_batch):
                resultats.append({
                    "fichier": nom_fichier,
                    "idee": idee,
                    "start": start_associe
                })
            textes_batch = []
            starts_batch = []

# --- Sauvegarde CSV ---
if resultats:
    df_idees = pd.DataFrame(resultats)
    df_idees.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\n📁 Idées sauvegardées dans {FICHIER_SORTIE} ✅")
else:
    print("\n⚠️ Aucune idée extraite.")
