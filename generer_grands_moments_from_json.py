import os
import json
import pandas as pd
import openai
from pathlib import Path

# --- Configuration ---
DOSSIER_JSON = "videos"
FICHIER_SORTIE = "grands_moments.csv"
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Fonction pour charger les segments d'un JSON ---
def charger_segments_json(fichier_json):
    with open(fichier_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])

# --- Fonction pour extraire plusieurs grands moments sur le texte complet ---
def extraire_grands_moments(texte):
    prompt = (
        "Voici la transcription complète d'une vidéo. Donne-moi entre 5 et 10 grands moments clés, "
        "sous forme de courtes phrases concrètes (exemples : 'Comment obtenir plus de mandats', 'Techniques pour répondre aux objections').\n\n"
        "Merci de répondre uniquement sous forme de liste numérotée.\n\n"
        f"Transcription :\n{texte}\n"
    )

    try:
        reponse = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        texte_brut = reponse.choices[0].message.content
        idees = []
        for ligne in texte_brut.split("\n"):
            if ligne.strip() and any(c.isalnum() for c in ligne):
                partie = ligne.split(".", 1)[-1].strip()
                if partie:
                    idees.append(partie)
        print(f"\n✅ Idées extraites : {idees}")
        return idees
    except Exception as e:
        print(f"\n❌ Erreur OpenAI : {e}")
        return []

# --- Traitement ---
resultats = []

for fichier in Path(DOSSIER_JSON).glob("*.json"):
    nom_fichier = fichier.stem
    print(f"\n✨ Traitement de : {nom_fichier}")

    segments = charger_segments_json(fichier)
    segments = [seg for seg in segments if seg.get("text", "").strip()]

    if not segments:
        print(f"⚠️ Pas de contenu utile pour {nom_fichier}, fichier ignoré.")
        continue

    full_text = " ".join([seg["text"] for seg in segments])

    if full_text.strip():
        idees = extraire_grands_moments(full_text)
        if idees:
            duree_totale = segments[-1]["end"] if segments else 0
            intervalle = duree_totale / len(idees) if idees else 0
            for idx, idee in enumerate(idees):
                start = int(idx * intervalle)
                resultats.append({
                    "fichier": nom_fichier,
                    "idee": idee,
                    "start": start
                })

# --- Sauvegarde CSV ---
if resultats:
    df_idees = pd.DataFrame(resultats)
    df_idees.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\n📁 Grands moments sauvegardés dans {FICHIER_SORTIE} ✅")
else:
else:
    print(f"⚠️ Aucune idée trouvée pour {nom_fichier}")
