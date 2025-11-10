import os
import csv
import openai
import pandas as pd
from pathlib import Path

# --- Paramètres ---
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
FICHIER_SORTIE = "themes.csv"

# Incrémentalité
# - FORCE_REBUILD=1   : régénère tous les thèmes (remplace les lignes existantes)
# - UPDATE_EXISTING=1 : ne traite que les fichiers déjà présents et les remplace
FORCE_REBUILD   = os.getenv("FORCE_REBUILD", "0") == "1"
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "0") == "1"

openai.api_key = os.getenv("OPENAI_API_KEY")

BAD_SEPARATORS = ["|", ";", ",", " / ", "/", "\\", "  "]

def sanitize_theme(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    for sep in BAD_SEPARATORS:
        if sep != "|":
            s = s.replace(sep, " ")
    s = " ".join(s.split())
    return s

def dedupe_preserve_order(items):
    seen = set(); out = []
    for x in items:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

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

    texte_brut = (resp.choices[0].message.content or "")
    themes = []
    for ligne in texte_brut.splitlines():
        ligne = ligne.strip()
        if not ligne: continue
        if "." in ligne[:3]:
            ligne = ligne.split(".", 1)[-1].strip()
        elif ")" in ligne[:3]:
            ligne = ligne.split(")", 1)[-1].strip()
        theme = sanitize_theme(ligne)
        if theme:
            themes.append(theme)
    themes = dedupe_preserve_order(themes)[:5]
    return themes

def _load_existing_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["fichier", "themes"])
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
    except Exception:
        df = pd.DataFrame(columns=["fichier", "themes"])
    for col in ("fichier","themes"):
        if col not in df.columns:
            df[col] = ""
    return df

def main():
    existing = _load_existing_csv(FICHIER_SORTIE)
    processed = set(existing["fichier"].astype(str)) if len(existing) else set()

    resultats = []
    fichiers = list(Path(DOSSIER_RESUME).glob("*.txt"))
    print(f"[INFO] Résumés trouvés: {len(fichiers)}")

    for fichier_txt in fichiers:
        nom_fichier = fichier_txt.stem

        if not (FORCE_REBUILD or UPDATE_EXISTING):
            if nom_fichier in processed:
                print(f"↪️ Déjà présent dans {FICHIER_SORTIE}: {nom_fichier} — saut.")
                continue

        texte = fichier_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not texte:
            print(f"⚠️ Résumé vide pour {nom_fichier}, saut.")
            continue

        texte = texte[:5000]
        print(f"✨ Extraction des thèmes pour : {nom_fichier}")
        themes = extraire_themes(texte)
        themes_join = "|".join(themes)
        resultats.append({"fichier": nom_fichier, "themes": themes_join})

    if not resultats:
        print("⚠️ Aucune nouveauté à écrire.")
        return

    df_new = pd.DataFrame(resultats, columns=["fichier", "themes"])

    if FORCE_REBUILD or UPDATE_EXISTING:
        mask_keep = ~existing["fichier"].isin(df_new["fichier"])
        df_final = pd.concat([existing[mask_keep], df_new], ignore_index=True)
    else:
        mask_new = ~df_new["fichier"].isin(processed)
        df_final = pd.concat([existing, df_new[mask_new]], ignore_index=True)

    df_final.drop_duplicates(subset=["fichier"], keep="last", inplace=True)
    df_final.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8", sep=",", quoting=csv.QUOTE_MINIMAL)
    nb_tags = sum(len((r or "").split("|")) for r in df_final["themes"])
    print(f"📁 Thèmes sauvegardés dans {FICHIER_SORTIE} ✅")
    print(f"→ {len(df_final)} fichiers au total, {nb_tags} tags cumulés.")

if __name__ == "__main__":
    main()
