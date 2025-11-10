# incremental_utils.py
# Utilitaires communs d'incrémentalité basés sur 2 mécanismes complémentaires :
# 1) "Sentinelles" par élément traité (un fichier .done par clé) -> rapide et robuste
# 2) Registre CSV central "traitements/registre.csv" pour audit / reprise / debug
#
# Usage typique (dans vos scripts) :
#   from incremental_utils import (
#       compute_fingerprint, should_skip, mark_done, mark_failed,
#       backfill_if_up_to_date
#   )
#
# Convention de clé :
# - newsletters : key = <stem du .txt>
# - idees_v2    : key = f"{stem}__start_{start}"
# - posts       : key = <stem du fichier vidéo ou du CSV urls>
#
# Variables d'env utiles :
# - TRAITEMENTS_DIR         (chemin)   : dossier racine du registre/sentinelles (par défaut "traitements")
# - FORCE_REBUILD=1                    : force le retraitement
# - UPDATE_EXISTING=1                  : ne traite QUE ce qui a déjà une sentinelle (utile pour refresh ciblé)
# - IGNORE_SENTINELS=1                 : ignore totalement les sentinelles (désactive le skip)
# - AUTO_BACKFILL_SENTINEL=1           : crée une sentinelle si la sortie est déjà à jour (par défaut ON)
#
from __future__ import annotations
import os, csv, hashlib, json, time, re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# --- Répertoires & fichiers ---
TRAITEMENTS_DIR = Path(os.getenv("TRAITEMENTS_DIR", "traitements"))
SENTINELS_DIR   = TRAITEMENTS_DIR / "sentinels"
REGISTRE_CSV    = TRAITEMENTS_DIR / "registre.csv"

# --- CSV registre ---
REG_HEADERS = ["script","key","source","fingerprint","output","status","updated_at","extra"]

def _ensure_dirs(script: str):
    (SENTINELS_DIR / script).mkdir(parents=True, exist_ok=True)
    TRAITEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRE_CSV.exists():
        with REGISTRE_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REG_HEADERS)
            w.writeheader()

def _now_iso():
    return datetime.now().isoformat(timespec="seconds")

def _safe_component(s: str) -> str:
    s = re.sub(r"[^\w\.-]+", "_", s, flags=re.S).strip("_")
    return s[:64] if len(s) > 64 else s

def _key_filename(key: str) -> str:
    base = _safe_component(key)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{base}__{digest}.done"

def sentinel_path(script: str, key: str) -> Path:
    return (SENTINELS_DIR / script / _key_filename(key))

def read_sentinel(script: str, key: str) -> Optional[str]:
    p = sentinel_path(script, key)
    if not p.exists():
        return None
    try:
        raw = p.read_text(encoding="utf-8").strip()
        return raw.splitlines()[0].strip() if raw else None
    except Exception:
        return None

def write_sentinel(script: str, key: str, fingerprint: str, extra: Optional[Dict]=None):
    _ensure_dirs(script)
    p = sentinel_path(script, key)
    body = fingerprint
    if extra:
        try:
            body += "\n" + json.dumps(extra, ensure_ascii=False)
        except Exception:
            pass
    p.write_text(body, encoding="utf-8")

def compute_fingerprint(*parts: Optional[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p is None: 
            continue
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\x00")  # séparateur
    return h.hexdigest()

def should_skip(script: str, key: str, fingerprint: str) -> bool:
    if os.getenv("FORCE_REBUILD", "0") == "1":
        return False
    if os.getenv("IGNORE_SENTINELS", "0") == "1":
        return False
    _ensure_dirs(script)
    sent = read_sentinel(script, key)
    if os.getenv("UPDATE_EXISTING", "0") == "1":
        # Mode 'refresh' : on ne traite que si une sentinelle existe déjà ET que la FP a changé
        return (sent is None) or (sent == fingerprint)
    # Mode normal : si la sentinelle matche -> skip
    return sent == fingerprint

def mark_done(script: str, key: str, source: str, fingerprint: str, output: str, extra: Optional[Dict]=None):
    write_sentinel(script, key, fingerprint, extra=extra)
    _ensure_dirs(script)
    rows = []
    if REGISTRE_CSV.exists():
        try:
            with REGISTRE_CSV.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
    # Upsert
    updated = False
    for r in rows:
        if r.get("script")==script and r.get("key")==key:
            r.update({
                "source": source, "fingerprint": fingerprint, "output": output,
                "status": "done", "updated_at": _now_iso(),
                "extra": json.dumps(extra, ensure_ascii=False) if extra else ""
            })
            updated = True
            break
    if not updated:
        rows.append({
            "script": script, "key": key, "source": source, "fingerprint": fingerprint,
            "output": output, "status":"done", "updated_at": _now_iso(),
            "extra": json.dumps(extra, ensure_ascii=False) if extra else ""
        })
    with REGISTRE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REG_HEADERS)
        w.writeheader()
        w.writerows(rows)

def mark_failed(script: str, key: str, source: str, fingerprint: str, output: str, error_msg: str):
    _ensure_dirs(script)
    rows = []
    if REGISTRE_CSV.exists():
        try:
            with REGISTRE_CSV.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
    rows.append({
        "script": script, "key": key, "source": source, "fingerprint": fingerprint,
        "output": output, "status":"failed", "updated_at": _now_iso(),
        "extra": json.dumps({"error": error_msg}, ensure_ascii=False)
    })
    with REGISTRE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REG_HEADERS)
        w.writeheader()
        w.writerows(rows)

def backfill_if_up_to_date(script: str, key: str, fingerprint: str, source_path: Path, output_path: Path) -> bool:
    """Si la sortie existe et est plus récente que la source, on crée la sentinelle (si manquante)."""
    if os.getenv("AUTO_BACKFILL_SENTINEL","1") != "1":
        return False
    if not output_path.exists() or not source_path.exists():
        return False
    try:
        if output_path.stat().st_mtime >= source_path.stat().st_mtime:
            if read_sentinel(script, key) != fingerprint:
                write_sentinel(script, key, fingerprint, extra={"up_to_date":"mtime"})
            return True
    except Exception:
        return False
    return False
