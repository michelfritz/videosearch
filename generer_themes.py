import os
import openai
import pandas as pd
from pathlib import Path

# --- Paramètres ---
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
FICHIER_SORTIE = "themes.csv"

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Fonction d'appel OpenAI pour extraire les thèmes courts ---
def extraire_themes(texte_resume):
    prompt = (
        "Voici le résumé d'une vidéo. Extrait entre 3 et 5 thèmes principaux très courts (2 à 4 mots maximum), "
        "sous forme de mots clés synthétiques, très concis, sans phrase complète. "
        "Formate la réponse sous forme d'une liste numérotée sans texte additionnel.\n\n"
        f"Texte :\n{texte_resume}\n"
    )

    reponse = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    texte_brut = reponse.choices[0].message.content
    themes = []
    for ligne in texte_brut.split("\n"):
        if ligne.strip():
            partie = ligne.split(".", 1)[-1].strip()
            if partie:
                themes.append(partie)
    return themes

# --- Lecture et traitement ---
resultats = []

for fichier_txt in Path(DOSSIER_RESUME).glob("*.txt"):
    nom_fichier = fichier_txt.stem
    print(f"\n✨ Extraction des thèmes pour : {nom_fichier}")

    with open(fichier_txt, "r", encoding="utf-8") as f:
        texte_resume = f.read()

    if texte_resume.strip():
        texte_resume = texte_resume[:5000]  # Limiter pour éviter dépassement de tokens
        themes = extraire_themes(texte_resume)
        resultats.append({
            "fichier": nom_fichier,
            "themes": " | ".join(themes)
        })
    else:
        print(f"⚠️ Résumé vide pour {nom_fichier}, saut.")

# --- Sauvegarde CSV ---
if resultats:
    df_themes = pd.DataFrame(resultats)
    df_themes.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\n📁 Thèmes sauvegardés dans {FICHIER_SORTIE} ✅")
else:
    print("\n⚠️ Aucun thème extrait.")
