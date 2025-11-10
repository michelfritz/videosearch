
import os
import csv
import openai
import pandas as pd
from pathlib import Path

# --- Paramètres ---
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
FICHIER_SORTIE = "themes.csv"

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Utils de normalisation ---
BAD_SEPARATORS = ["|", ";", ",", " / ", "/", "\\", "  "]

def sanitize_theme(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # Éviter les séparateurs parasites à l'intérieur d'un tag
    for sep in BAD_SEPARATORS:
        if sep != "|":  # on garde '|' uniquement pour la jointure finale
            s = s.replace(sep, " ")
    # Nettoyages
    s = " ".join(s.split())  # compresser espaces
    return s

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

# --- Appel OpenAI pour extraire les thèmes courts ---
def extraire_themes(texte_resume):
    prompt = (
        "Voici le résumé d'une vidéo. Extrait entre 3 et 5 thèmes principaux très courts (2 à 4 mots max), "
        "sous forme de mots clés synthétiques, très concis, sans phrase complète. "
        "Ces thèmes doivent être pertinents pour l'immobilier ou le business (ignore le hors-sujet). "
        "Réponds sous forme de liste numérotée, sans autre texte.\n\n"
        f"Texte :\n{texte_resume}\n"
    )

    resp = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    texte_brut = resp.choices[0].message.content or ""
    themes = []
    for ligne in texte_brut.splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        # retirer le numéro "1." "2)" etc.
        if "." in ligne[:3]:
            ligne = ligne.split(".", 1)[-1].strip()
        elif ")" in ligne[:3]:
            ligne = ligne.split(")", 1)[-1].strip()
        theme = sanitize_theme(ligne)
        if theme:
            themes.append(theme)
    # dédoublonner et tronquer à 3-5 éléments
    themes = dedupe_preserve_order(themes)[:5]
    return themes

# --- Lecture et traitement ---
resultats = []

for fichier_txt in Path(DOSSIER_RESUME).glob("*.txt"):
    nom_fichier = fichier_txt.stem
    print(f"✨ Extraction des thèmes pour : {nom_fichier}")

    with open(fichier_txt, "r", encoding="utf-8") as f:
        texte = f.read()

    if not texte.strip():
        print(f"⚠️ Résumé vide pour {nom_fichier}, saut.")
        continue

    texte = texte[:5000]  # limite sécurité
    themes = extraire_themes(texte)
    # jointure stricte avec '|' SANS espaces autour (attendu par l'app)
    themes_join = "|".join(themes)
    resultats.append({"fichier": nom_fichier, "themes": themes_join})

# --- Sauvegarde CSV avec entêtes propres ---
if resultats:
    df = pd.DataFrame(resultats, columns=["fichier", "themes"])
    # Forcer exactement ces colonnes et empêcher toute pollution d'en-tête
    df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8", sep=",", quoting=csv.QUOTE_MINIMAL)
    print(f"📁 Thèmes sauvegardés dans {FICHIER_SORTIE} ✅")
    # Petit récap
    nb_tags = sum(len((r["themes"] or "").split("|")) for r in resultats)
    print(f"→ {len(df)} fichiers traités, {nb_tags} tags générés.")
else:
    print("⚠️ Aucun thème extrait.")
