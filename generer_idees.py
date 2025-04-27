import os
import openai
import pandas as pd
from pathlib import Path

# --- Paramètres ---
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
FICHIER_SORTIE = "idees.csv"

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Fonction d'appel OpenAI pour extraire les idées ---
def extraire_idees(texte_resume):
    prompt = (
        "Voici le résumé d'une vidéo. Extrait entre 5 et 10 idées principales concrètes et actionnables, "
        "sous forme de courtes phrases claires adaptées à l'immobilier ou au business. "
        "Formate la réponse sous forme d'une liste numérotée sans texte additionnel.\n\n"
        f"Texte :\n{texte_resume}\n"
    )

    reponse = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
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

# --- Lecture et traitement ---
resultats = []

for fichier_txt in Path(DOSSIER_RESUME).glob("*.txt"):
    nom_fichier = fichier_txt.stem
    print(f"\n✨ Extraction des idées pour : {nom_fichier}")

    with open(fichier_txt, "r", encoding="utf-8") as f:
        texte_resume = f.read()

    if texte_resume.strip():
        idees = extraire_idees(texte_resume)
        resultats.append({
            "fichier": nom_fichier,
            "idees": " | ".join(idees)
        })
    else:
        print(f"⚠️ Résumé vide pour {nom_fichier}, saut.")

# --- Sauvegarde CSV ---
if resultats:
    df_idees = pd.DataFrame(resultats)
    df_idees.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\n📁 Idées sauvegardées dans {FICHIER_SORTIE} ✅")
else:
    print("\n⚠️ Aucune idée extraite.")
