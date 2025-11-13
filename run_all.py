#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_orchestrateur_tout.py — Orchestrateur séquentiel (incrémental)
# Étapes (dans le même dossier) :
#   1) transcrire_et_decouper.py         (déjà raisonnablement incrémental)
#   2) fusionner_et_vectoriser_inc.py    (vectorisation incrémentale + cache)
#      (fallback: fusionner_et_vectoriser.py si _inc absent)
#   3) interrogation_inc.py              (FAISS incrémental, ajout des nouveaux)
#      (fallback: interrogation.py si _inc absent)
#   4) generer_idees_v2.py               (réparé, incrémental par sentinelles)
#   5) generer_idees.py                  (sentinelles)
#   6) generer_themes.py                 (sentinelles)
#   7) generer_newsletters.py            (sentinelles + backfill mtime)
#   8) generer_posts_sociaux.py          (sentinelles + backfill si draft complet)
#
# Variables utiles :
#   OPENAI_API_KEY, TRAITEMENTS_DIR, FORCE_REBUILD, UPDATE_EXISTING, etc.

import sys, os, time, subprocess
from pathlib import Path

def banner(msg: str):
    line = "=" * 96
    print(f"\n{line}\n{msg}\n{line}", flush=True)

def run_cmd(args, cwd: Path):
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cp = subprocess.run(args, cwd=str(cwd), env=env)
    if cp.returncode != 0:
        raise SystemExit(f"[ERREUR] Commande échouée (code {cp.returncode}) : {' '.join(args)}")

def main():
    base = Path(__file__).resolve().parent
    py = sys.executable or "python"

    steps = [
        ("Transcrire & découper",   ["transcrire_et_decouper.py"]),
        ("Fusionner + vectoriser",  ["fusionner_et_vectoriser_inc.py", "fusionner_et_vectoriser.py"]),
        ("Index FAISS",             ["interrogation_inc.py", "interrogation.py"]),
        ("Idées V2 (blocs)",        ["generer_idees_v2.py"]),
        ("Idées (résumés)",         ["generer_idees.py"]),
        ("Thèmes",                  ["generer_themes.py"]),
        ("Newsletters",             ["generer_newsletters.py"]),
        ("Posts sociaux",           ["generer_posts_sociaux.py"]),
    ]

    t0 = time.time()
    for label, candidates in steps:
        # Résoudre le script à lancer (prend le premier existant)
        script = None
        for c in candidates:
            if (base / c).exists():
                script = c
                break
        if script is None:
            print(f"[SKIP] {label} — aucun des fichiers n'existe: {candidates}")
            continue

        banner(f"[RUN] {label} — {script}")
        run_cmd([py, "-u", script], cwd=base)
        print(f"[OK] {label}", flush=True)

    dt = time.time() - t0
    banner(f"[OK] Orchestration terminée en {dt/60:.1f} min.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur.")
