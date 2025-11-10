import os
import pandas as pd
import openai
from pathlib import Path

# --- Paramètres ---
DOSSIER_BLOCS = "blocs"
FICHIER_SORTIE = "idees_v2.csv"

# Incrémentalité via env
# - FORCE_REBUILD=1   : régénérer toutes les idées (remplace les paires existantes)
# - UPDATE_EXISTING=1 : ne traite que les paires déjà présentes et les remplace
FORCE_REBUILD   = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

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
        return pd.DataFrame(columns=["fichier", "idee", "start"])
    try:
        df = pd.read_csv(p, dtype={"fichier": str, "idee": str, "start": int}, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["fichier", "idee", "start"])
    for col in ("fichier","idee","start"):
        if col not in df.columns:
            df[col] = "" if col != "start" else 0
    return df

def main():
    existing = _load_existing_csv(FICHIER_SORTIE)
    existing["fichier"] = existing["fichier"].astype(str)
    # ensemble des paires (fichier,start) déjà présentes
    done_pairs = set((str(r["fichier"]), int(r["start"])) for _, r in existing.iterrows())

    resultats = []
    fichiers_csv = list(Path(DOSSIER_BLOCS).glob("*.csv"))
    print(f"[INFO] Fichiers de blocs trouvés: {len(fichiers_csv)}")

    for fichier_csv in fichiers_csv:
        nom_fichier = fichier_csv.stem
        print(f"\n✨ Traitement de : {nom_fichier}")

        blocs_df = pd.read_csv(fichier_csv)
        textes_batch, starts_batch = [], []

        for idx, row in blocs_df.iterrows():
            texte_bloc = str(row.get("text", "")).strip()
            try:
                start = int(row.get("start", 0))
            except Exception:
                start = 0

            pair = (nom_fichier, start)

            # Éviter retraitements
            if not (FORCE_REBUILD or UPDATE_EXISTING):
                if pair in done_pairs:
                    continue

            if texte_bloc:
                textes_batch.append(texte_bloc)
                starts_batch.append(start)

            # Envoi par paquets de 10 (ou fin de fichier)
            if len(textes_batch) == 10 or (idx == len(blocs_df) - 1 and textes_batch):
                idees_batch = extraire_idees_batch(textes_batch)
                # zip court : ignore les extra si mismatch
                for idee, start_associe in zip(idees_batch, starts_batch):
                    resultats.append({
                        "fichier": nom_fichier,
                        "idee": idee,
                        "start": int(start_associe)
                    })
                textes_batch, starts_batch = [], []

    if not resultats:
        print("\n⚠️ Aucune nouveauté à écrire.")
        return

    df_new = pd.DataFrame(resultats)

    if FORCE_REBUILD or UPDATE_EXISTING:
        # Remplace les paires existantes (fichier,start)
        key_cols = ["fichier","start"]
        existing_key = set(map(tuple, existing[key_cols].astype({"fichier":str,"start":int}).values.tolist()))
        # On garde les anciennes lignes qui ne sont pas dans df_new
        mask_keep = ~existing.apply(lambda r: (str(r["fichier"]), int(r["start"])) in set(map(tuple, df_new[key_cols].astype({"fichier":str,"start":int}).values.tolist())), axis=1)
        df_final = pd.concat([existing[mask_keep], df_new], ignore_index=True)
    else:
        # Ajoute uniquement les nouvelles paires
        existing_key = set((str(r["fichier"]), int(r["start"])) for _, r in existing.iterrows())
        df_new = df_new[~df_new.apply(lambda r: (str(r["fichier"]), int(r["start"])) in existing_key, axis=1)]
        df_final = pd.concat([existing, df_new], ignore_index=True)

    df_final.drop_duplicates(subset=["fichier","start"], keep="last", inplace=True)
    df_final.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\n📁 Idées V2 mises à jour dans {FICHIER_SORTIE} ✅ ({len(df_final)} lignes au total)")

if __name__ == "__main__":
    main()
