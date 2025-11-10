import os
import pandas as pd
import openai
from pathlib import Path
from incremental_utils import compute_fingerprint, should_skip, mark_done

# --- Paramètres ---
DOSSIER_BLOCS = os.getenv("DOSSIER_BLOCS", "blocs")
FICHIER_SORTIE = os.getenv("FICHIER_SORTIE", "idees_v2.csv")

# Contrôles standards (reconnus par incremental_utils)
FORCE_REBUILD   = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

openai.api_key = os.getenv("OPENAI_API_KEY")

SCRIPT_NAME = "idees_v2"

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
    fichiers_csv = list(Path(DOSSIER_BLOCS).glob("*.csv"))
    print(f"[INFO] Fichiers de blocs trouvés: {len(fichiers_csv)}")

    to_append = []

    for fichier_csv in fichiers_csv:
        nom_fichier = fichier_csv.stem
        print(f"\n✨ {SCRIPT_NAME}: {nom_fichier}")

        try:
            blocs_df = pd.read_csv(fichier_csv)
        except Exception as e:
            print(f"[WARN] Lecture impossible: {fichier_csv} ({e})")
            continue

        textes_batch, starts_batch, keys_batch, fps_batch = [], [], [], []

        for idx, row in blocs_df.iterrows():
            texte_bloc = str(row.get("text", "")).strip()
            try:
                start = int(row.get("start", 0))
            except Exception:
                start = 0

            key = f"{nom_fichier}__start_{start}"
            fp  = compute_fingerprint(nom_fichier, start, texte_bloc)

            if not FORCE_REBUILD and should_skip(SCRIPT_NAME, key, fp):
                # déjà traité avec le même contenu
                continue

            if not texte_bloc:
                continue

            textes_batch.append(texte_bloc)
            starts_batch.append(start)
            keys_batch.append(key)
            fps_batch.append(fp)

            # Paquet de 10 ou fin
            if len(textes_batch) == 10 or (idx == len(blocs_df) - 1 and textes_batch):
                idees = extraire_idees_batch(textes_batch)
                for idee, start_associe, key_i, fp_i in zip(idees, starts_batch, keys_batch, fps_batch):
                    to_append.append({"fichier": nom_fichier, "idee": idee, "start": int(start_associe)})
                    # Marque comme fait (sentinelle + registre)
                    mark_done(SCRIPT_NAME, key_i, str(fichier_csv), fp_i, FICHIER_SORTIE, extra={"idee": idee})
                # Reset batch
                textes_batch, starts_batch, keys_batch, fps_batch = [], [], [], []

    if not to_append:
        print("\n⚠️ Aucune nouveauté à écrire.")
        return

    df_new = pd.DataFrame(to_append, columns=["fichier","idee","start"])

    if FORCE_REBUILD or UPDATE_EXISTING:
        existing_key = set(map(tuple, existing[["fichier","start"]].astype({"fichier":str,"start":int}).values.tolist()))
        mask_keep = ~existing.apply(lambda r: (str(r["fichier"]), int(r["start"])) in set(map(tuple, df_new[["fichier","start"]].astype({"fichier":str,"start":int}).values.tolist())), axis=1)
        df_final = pd.concat([existing[mask_keep], df_new], ignore_index=True)
    else:
        existing_key = set((str(r["fichier"]), int(r["start"])) for _, r in existing.iterrows())
        df_new = df_new[~df_new.apply(lambda r: (str(r["fichier"]), int(r["start"])) in existing_key, axis=1)]
        df_final = pd.concat([existing, df_new], ignore_index=True)

    df_final.drop_duplicates(subset=["fichier","start"], keep="last", inplace=True)
    Path(FICHIER_SORTIE).write_text(df_final.to_csv(index=False, encoding="utf-8"), encoding="utf-8")
    print(f"\n📁 Idées V2 à jour -> {FICHIER_SORTIE} ({len(df_final)} lignes)")

if __name__ == "__main__":
    main()
