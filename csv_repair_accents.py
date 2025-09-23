
"""
csv_repair_accents.py
---------------------
Outil de RÉPARATION ponctuelle de votre CSV actuel :
- Renomme la colonne 'ï»¿fichier' en 'fichier' (BOM affiché) et supprime tout BOM caché
- Tente de corriger la "mojibake" typique (ex: 'RÃ©ponses' -> 'Réponses') sur la colonne 'fichier'
- NE TOUCHE PAS aux autres colonnes, sauf à enlever un BOM éventuel au début
- Ecrit un .backup avant de réécrire le CSV

Usage :
  python csv_repair_accents.py --csv_path "C:\\Transcript\\urls.csv" --out_encoding "utf-8-sig"
"""

import argparse
import pandas as pd

BOM = "\ufeff"

def unmojibake(s: str) -> str:
    """
    Heuristique: si la chaîne contient 'Ã' ou 'Â', on tente un roundtrip latin1->utf-8.
    Cela répare souvent les cas 'RÃ©ponses' -> 'Réponses'.
    """
    if not isinstance(s, str):
        return s
    if ("Ã" in s) or ("Â" in s):
        try:
            return s.encode("latin1").decode("utf-8")
        except Exception:
            return s
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="Chemin du CSV à réparer")
    ap.add_argument("--out_encoding", default="utf-8-sig", help="Encodage de sortie (défaut utf-8-sig)")
    args = ap.parse_args()

    # 1) Lecture tolérante et nettoyage BOM colonnes
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(args.csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError("Impossible de lire le CSV en utf-8-sig/utf-8/cp1252.")

    # Nettoyage noms de colonnes et BOM en cellules
    new_cols = []
    for c in df.columns:
        c2 = str(c).replace(BOM, "").strip()
        # normaliser 'ï»¿fichier' -> 'fichier' si jamais
        if c2.lower() == "ï»¿fichier":
            c2 = "fichier"
        new_cols.append(c2)
    df.columns = new_cols

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).map(lambda x: x.replace(BOM, "").strip())

    # 2) Si la colonne 'fichier' existe, tenter la réparation "mojibake" sur ses valeurs
    if "fichier" in df.columns:
        df["fichier"] = df["fichier"].astype(str).map(unmojibake)

    # 3) Sauvegardes
    bak = args.csv_path + ".pre_repair.bak"
    df.to_csv(bak, index=False, encoding=args.out_encoding)
    df.to_csv(args.csv_path, index=False, encoding=args.out_encoding)
    print("✅ Réparation appliquée.")
    print("💾 Copie de sauvegarde :", bak)
    print("📝 Fichier mis à jour  :", args.csv_path)

if __name__ == "__main__":
    main()
