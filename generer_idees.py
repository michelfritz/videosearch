import os
import openai
import pandas as pd
from pathlib import Path

# --- Paramètres ---
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
FICHIER_SORTIE = "idees.csv"

# Contrôles d'incrémentalité (via variables d'environnement)
# - FORCE_REBUILD=1        : régénère TOUT (écrasera les lignes existantes pour les mêmes fichiers)
# - UPDATE_EXISTING=1      : ne traite QUE les fichiers présents ET remplace leurs lignes
#   (par défaut, on ne traite que les NOUVEAUX fichiers non encore présents dans le CSV)
FORCE_REBUILD   = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Fonction d'appel OpenAI pour extraire les idées ---
def extraire_idees(texte_resume: str):
    prompt = (
        "Voici le résumé d'une vidéo. Extrait entre 5 et 10 idées principales concrètes et actionnables, "
        "sous forme de courtes phrases claires, dont il FAUT qu'elles soient adaptées à l'immobilier ou au business. "
        "Oublie tout les sujets hors du champs business et du secteur immobilier."
        "Formate la réponse sous forme d'une liste numérotée sans texte additionnel.\n\n"
        f"Texte :\n{texte_resume}\n"
    )

    reponse = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    texte_brut = reponse.choices[0].message.content or ""
    idees = []
    for ligne in texte_brut.split("\n"):
        if ligne.strip():
            partie = ligne.split(".", 1)[-1].strip()
            if partie:
                idees.append(partie)
    return idees

def _load_existing_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["fichier", "idees"])
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["fichier", "idees"])
    if "fichier" not in df.columns:
        df["fichier"] = ""
    if "idees" not in df.columns:
        df["idees"] = ""
    df["fichier"] = df["fichier"].astype(str)
    df["idees"] = df["idees"].astype(str)
    return df

def main():
    existing = _load_existing_csv(FICHIER_SORTIE)
    processed = set(existing["fichier"].astype(str))

    fichiers = list(Path(DOSSIER_RESUME).glob("*.txt"))
    print(f"[INFO] Résumés trouvés: {len(fichiers)}")
    resultats_new = []

    for fichier_txt in fichiers:
        nom_fichier = fichier_txt.stem
        # Stratégie d'évitement des retraitements
        if not (FORCE_REBUILD or UPDATE_EXISTING):
            if nom_fichier in processed:
                print(f"↪️ Déjà présent dans {FICHIER_SORTIE}: {nom_fichier} — saut.")
                continue

        texte_resume = fichier_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not texte_resume:
            print(f"⚠️ Résumé vide pour {nom_fichier}, saut.")
            continue

        print(f"✨ Extraction des idées pour : {nom_fichier}")
        idees = extraire_idees(texte_resume)
        resultats_new.append({"fichier": nom_fichier, "idees": " | ".join(idees)})

    if not resultats_new:
        print("⚠️ Aucune nouveauté à écrire.")
        return

    df_new = pd.DataFrame(resultats_new)

    if FORCE_REBUILD or UPDATE_EXISTING:
        # Remplacer les lignes existantes ayant le même 'fichier'
        mask_keep = ~existing["fichier"].isin(df_new["fichier"])
        df_final = pd.concat([existing[mask_keep], df_new], ignore_index=True)
    else:
        # Ajouter uniquement les nouveaux fichiers
        mask_new = ~df_new["fichier"].isin(processed)
        df_final = pd.concat([existing, df_new[mask_new]], ignore_index=True)

    df_final.drop_duplicates(subset=["fichier"], keep="last", inplace=True)
    df_final.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\n📁 Idées mises à jour dans {FICHIER_SORTIE} ✅ ({len(df_final)} lignes au total)")

if __name__ == "__main__":
    main()
