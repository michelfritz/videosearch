# generer_posts_sociaux.py — Sentinelles + registre central
import os, re, json, base64, urllib.parse, hashlib, random, time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np, cv2

from incremental_utils import compute_fingerprint, should_skip, mark_done

load_dotenv()

# --- Incrémentalité ---
SKIP_IF_COMPLETE = os.getenv("SKIP_IF_COMPLETE", "1") == "1"

# --- Configuration ---
URLS_CSV   = os.getenv("URLS_CSV", r"C:\Transcript\urls.csv")
RESUME_DIR = Path(os.getenv("RESUME_DIR", r"C:\Transcript\Dropbox (Personal)\resume"))
DRAFTS_DIR = Path(os.getenv("DRAFTS_DIR", "social_drafts")); DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS        = int(os.getenv("MAX_ROWS", "0"))                 # 0 = tout
IMAGE_PROVIDER  = os.getenv("IMAGE_PROVIDER", "hybrid").lower()   # hybrid | openai | unsplash
IMAGE_MODEL     = os.getenv("IMAGE_MODEL", "gpt-image-1")
OPENAI_QUALITY  = os.getenv("OPENAI_QUALITY", "high")             # high|medium|low|auto

SCRIPT_NAME = "posts_sociaux"

# --- Unsplash API ---
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
UNSPLASH_PER_PAGE   = int(os.getenv("UNSPLASH_PER_PAGE", "30"))
UNSPLASH_API_BASE   = "https://api.unsplash.com"

# OpenAI client (optionnel)
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# ---------------------- Utils ----------------------
def _clean_filename(name: str) -> str:
    name = re.sub(r"[^\w\-\s\.]", "_", name).strip()
    return re.sub(r"\s+", "_", name)[:80]

def _read_resume_for_file(stem: str) -> str:
    if not RESUME_DIR.exists(): return ""
    cands = list(RESUME_DIR.glob("*.txt"))
    stem_low = stem.lower()
    exact = [p for p in cands if stem_low in p.stem.lower() or p.stem.lower() in stem_low]
    if exact:
        try: return exact[0].read_text(encoding="utf-8", errors="ignore")
        except Exception: return exact[0].read_text(errors="ignore")
    best, best_score = None, 0
    for p in cands:
        s = p.stem.lower()
        score = len(set(stem_low.split()) & set(s.split()))
        if score > best_score:
            best, best_score = p, score
    if best:
        try: return best.read_text(encoding="utf-8", errors="ignore")
        except Exception: return best.read_text(errors="ignore")
    return ""

def _public_title(title: str) -> str:
    t = title or ""
    t = re.sub(r"(?i)\bwebinar|webinaire|visi[oô]|formation\s+interne|atelier|module|newsletter|youtube|vid[ée]o", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -–—:·").strip()
    return t or "Actu immobilière"

def _parse_yt_id(url: str) -> str:
    if not url: return ""
    m = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else ""

def yt_thumb_url(video_id: str) -> str:
    return f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"

def download_with_retry(url: str, tries: int = 3, timeout: int = 20, headers: Optional[dict]=None) -> Optional[bytes]:
    if not url: return None
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers or {})
            if r.status_code == 200 and r.content:
                return r.content
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5)
    print(f"[WARN] download failed for {url} ({last_err})")
    return None

# -------------- recadrage conscient visage --------------
def _face_aware_crop(im: Image.Image, target_w: int, target_h: int) -> Optional[Image.Image]:
    try:
        gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x + w//2, y + int(h*0.45)
        tgt_ratio = target_w / target_h
        if im.width / im.height > tgt_ratio:
            crop_h = min(im.height, max(h*3, int(im.width / tgt_ratio)))
            crop_w = int(crop_h * tgt_ratio)
        else:
            crop_w = min(im.width, max(w*2, int(im.height * tgt_ratio)))
            crop_h = int(crop_w / tgt_ratio)
        x0 = max(0, min(im.width  - crop_w,  cx - crop_w//2))
        y0 = max(0, min(im.height - crop_h, cy - crop_h//2))
        im = im.crop((x0, y0, x0+crop_w, y0+crop_h))
        return im.resize((target_w, target_h), Image.LANCZOS)
    except Exception:
        return None

def save_resized(img_bytes: bytes, out_path: Path, target_w: int, target_h: int, focus_top: float=0.33, jitter: float=0.05):
    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    face = _face_aware_crop(im, target_w, target_h)
    if face is None:
        im_ratio = im.width / im.height
        tgt_ratio = target_w / target_h
        if im_ratio > tgt_ratio:
            new_h = im.height; new_w = int(new_h * tgt_ratio)
            x0 = (im.width - new_w)//2
            im = im.crop((x0, 0, x0+new_w, new_h))
        else:
            new_w = im.width; new_h = int(new_w / tgt_ratio)
            y0 = max(0, min(im.height-new_h, int((im.height-new_h)*focus_top)))
            im = im.crop((0, y0, new_w, y0+new_h))
        im = im.resize((target_w, target_h), Image.LANCZOS)
    else:
        im = face
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=92)

# ---------------------- Thématiques & génération captions (identique) ----------------------
# (conserve votre logique existante pour les topics/hashtags)
# ... pour compacité nous réutilisons votre code précédent quasiment tel quel ...
TOPIC_RULES = [
    {"key":"taux_credit","label":"Taux & crédit immobilier","match":[r"\btaux", r"cr[ée]dit", r"emprunt", r"banque"],
     "hashtags":["#taux","#creditimmobilier","#financement","#achatimmobilier","#immobilier","#banque","#emprunt"]},
    {"key":"dpe_renovation","label":"DPE & rénovation énergétique","match":[r"\bdpe\b", r"r[ée]novation", r"isolati"],
     "hashtags":["#DPE","#rénovation","#énergie","#logement","#transition","#immobilier","#travaux"]},
    {"key":"prix_transactions","label":"Prix au m² & volumes de transactions","match":[r"prix", r"€/m", r"m²", r"transactions?"],
     "hashtags":["#priximmobilier","#marchéimmobilier","#transactions","#notaires","#baromètre","#m2","#tendance"]},
]
def _extract_topics(text: str, title: str) -> List[dict]:
    base = (title or "") + "\n" + (text or "")
    scores = []
    for rule in TOPIC_RULES:
        score = 0
        for pat in rule["match"]:
            score += len(re.findall(pat, base, flags=re.I))
        if score > 0:
            scores.append({"key": rule["key"], "score": score, "rule": rule})
    scores.sort(key=lambda x: (-x["score"], x["key"]))
    return scores

def _sanitize_b2c_text(s: str) -> str:
    if not s: return s
    s = re.sub(r"https?://\S+", "", s)
    banned = r"(webinar|webinaire|visi[oô]|formation\s+interne|atelier|module|newsletter|youtube|vid[ée]o)"
    lines = [ln for ln in s.splitlines() if not re.search(banned, ln, flags=re.I)]
    s = "\n".join(lines)
    s = re.sub(r"\bagents?\s+immobiliers?\b", "conseillers immobiliers", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _topic_hashtags(key: str, n_ig:int=10, n_fb:int=5) -> Tuple[List[str], List[str]]:
    base = ["#immobilier", "#conseilsimmobilier"]
    for r in TOPIC_RULES:
        if r["key"] == key:
            base = list(dict.fromkeys(r["hashtags"] + ["#immobilier"]))
            break
    ig = (base + ["#réseau", "#actualité", "#france"])[:max(8, min(n_ig, 12))]
    fb = (base + ["#actualité"])[:max(3, min(n_fb, 6))]
    return ig, fb

def build_captions(text: str, title_hint: str) -> Dict[str, List[str]]:
    topics = _extract_topics(text, title_hint)
    key = topics[0]["key"] if topics else "prix_transactions"
    ig_tags, fb_tags = _topic_hashtags(key)
    base = _sanitize_b2c_text(title_hint or "Le point du marché")
    igA = f"{base}\n\n{_sanitize_b2c_text(text[:220])}\n\n" + " ".join(ig_tags)
    igB = f"Zoom sur {base.lower()}\n\n{_sanitize_b2c_text(text[220:440])}\n\n" + " ".join(ig_tags)
    fbA = f"{base}\n\n{_sanitize_b2c_text(text[:220])}\n\n" + " ".join(fb_tags)
    fbB = f"À retenir — {base}\n\n{_sanitize_b2c_text(text[220:440])}\n\n" + " ".join(fb_tags)
    return {"ig":[igA, igB], "fb":[fbA, fbB], "title": (base[:80] or "À La Lucarne")}

def _is_complete_draft(root: Path) -> bool:
    needed = ["meta.json", "ig_caption_A.txt","ig_caption_B.txt","fb_caption_A.txt","fb_caption_B.txt"]
    if not all((root/p).exists() for p in needed):
        return False
    try:
        meta = json.loads((root/"meta.json").read_text(encoding="utf-8"))
        def ok(node): return bool(node.get("candidates"))
        return ok(meta["ig"]["A"]) and ok(meta["ig"]["B"]) and ok(meta["fb"]["A"]) and ok(meta["fb"]["B"])
    except Exception:
        return False

def main():
    df = pd.read_csv(URLS_CSV, encoding="utf-8")
    if "fichier" not in df.columns:
        raise RuntimeError("La colonne 'fichier' est requise dans urls.csv")
    if "url" not in df.columns:
        df["url"] = ""
    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS)

    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Drafts"):
        fichier = str(row["fichier"])
        stem = Path(fichier).stem
        video_url = str(row.get("url", "")).strip()

        title_hint = re.sub(r"[_\-]+", " ", stem).strip().title()
        resume_text = _read_resume_for_file(stem)

        # Empreinte : résume + url (si elle change -> régénération)
        fp = compute_fingerprint(stem, resume_text, video_url)
        key = stem

        droot = DRAFTS_DIR / _clean_filename(stem)
        (droot / "images").mkdir(parents=True, exist_ok=True)
        meta_path = droot / "meta.json"

        # Backfill sentinelle si un brouillon complet est déjà présent
        if SKIP_IF_COMPLETE and _is_complete_draft(droot):
            from incremental_utils import write_sentinel
            write_sentinel(SCRIPT_NAME, key, fp, extra={"backfill":"complete_draft"})
            skipped += 1
            continue

        if should_skip(SCRIPT_NAME, key, fp):
            skipped += 1
            continue

        # Générer légendes
        caps = build_captions(resume_text, title_hint)
        def _write(path: Path, content: str): path.write_text(content.strip(), encoding="utf-8")
        _write(droot / "ig_caption_A.txt", caps["ig"][0])
        _write(droot / "ig_caption_B.txt", caps["ig"][1])
        _write(droot / "fb_caption_A.txt", caps["fb"][0])
        _write(droot / "fb_caption_B.txt", caps["fb"][1])

        # Images (placeholder léger : miniature YouTube si dispo)
        cands = {"ig":{"A":{"candidates":[]},"B":{"candidates":[]}},"fb":{"A":{"candidates":[]},"B":{"candidates":[]}}}
        vid = _parse_yt_id(video_url)
        if vid:
            thumb = f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg"
            img_bytes = download_with_retry(thumb)
            if img_bytes:
                from PIL import Image
                def _save(img_bytes, name, w, h):
                    out = droot/"images"/name; save_resized(img_bytes, out, w, h, 0.33, 0.05); return out
                _save(img_bytes, "IG_A_youtube.jpg", 1080, 1350)
                _save(img_bytes, "FB_A_youtube.jpg", 1200, 630)
                cands["ig"]["A"]["candidates"].append({"source":"youtube","image_local":"images/IG_A_youtube.jpg","image_url":thumb})
                cands["fb"]["A"]["candidates"].append({"source":"youtube","image_local":"images/FB_A_youtube.jpg","image_url":thumb})

        meta = {
            "video_file": fichier,
            "video_url": video_url,
            "title": caps["title"],
            "ig": {
                "A": {"caption_file": "ig_caption_A.txt", **cands["ig"]["A"]},
                "B": {"caption_file": "ig_caption_B.txt", **cands["ig"]["B"]},
            },
            "fb": {
                "A": {"caption_file": "fb_caption_A.txt", **cands["fb"]["A"]},
                "B": {"caption_file": "fb_caption_B.txt", **cands["fb"]["B"]},
            },
            "provider": "hybrid",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Marque comme fait
        mark_done(SCRIPT_NAME, key, f"{URLS_CSV}", fp, str(droot))

    print(f"\n[OK] Brouillons à jour dans: {DRAFTS_DIR.resolve()} — ignorés: {skipped}")

if __name__ == "__main__":
    main()
