#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrateur séquentiel : transcrire -> fusionner & vectoriser -> interrogation
Exécute les scripts existants, sans les modifier, dans CE MÊME RÉPERTOIRE.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

STEPS = [
    ("Transcription / découpages / SRT / urls.csv",
     "transcrire_et_decouper.py"),
    ("Fusion des blocs + vectorisation (embeddings)",
     "fusionner_et_vectoriser.py"),
    ("Construction de l’index FAISS (interrogation)",
     "interrogation.py"),
]

def banner(msg: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{msg}\n{line}", flush=True)

def run_cmd(args, cwd: Path) -> None:
    banner(f"[RUN] {' '.join(args)}")
    # Débloque l’affichage temps-réel des sous‑processus
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cp = subprocess.run(args, cwd=str(cwd), env=env)
    if cp.returncode != 0:
        raise SystemExit(f"[ERREUR] Commande échouée (code {cp.returncode}) : {' '.join(args)}")

def main() -> None:
    base = Path(__file__).resolve().parent
    python_exe = sys.executable or "python"

    # Vérification présence des scripts
    missing = [s for _, s in STEPS if not (base / s).exists()]
    if missing:
        raise SystemExit(f"[ERREUR] Fichiers manquants dans {base} : {', '.join(missing)}")

    # Avertissements non bloquants (les scripts s’en chargeront aussi)
    if shutil.which("ffmpeg") is None:
        print("[AVERTISSEMENT] ffmpeg n’est pas détecté dans le PATH ; "
              "le script de transcription en aura besoin.", flush=True)
    if not os.environ.get("OPENAI_API_KEY"):
        print("[AVERTISSEMENT] OPENAI_API_KEY n’est pas défini ; "
              "la vectorisation/FAISS risque d’échouer.", flush=True)

    t0 = time.time()
    for label, script in STEPS:
        t_step = time.time()
        run_cmd([python_exe, "-u", script], cwd=base)
        dt = time.time() - t_step
        print(f"[OK] Étape « {label} » terminée en {dt:.1f}s.", flush=True)

    total = time.time() - t0
    banner(f"[OK] Pipeline complet terminé en {total/60:.1f} min.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur.")
