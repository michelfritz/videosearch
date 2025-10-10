# generer_posts_sociaux.py — Brouillons “flex” (base: version prod)
# - 4 candidats image par variante (IG A/B, FB A/B) : 3 Unsplash + 1 OpenAI (si dispo)
# - Recadrage visage + fallback robustes (Unsplash→OpenAI)
# - Skip idempotent si brouillon complet existe déjà
# - meta.json contient candidates[] et selected (par variante)
# - Compat descendante : les captions .txt sont conservées ; les images finales (IG_A.jpg, etc.)
#   seront choisies/figées dans l’interface (social_studio.py) via le candidat sélectionné.

import os, re, json, base64, urllib.parse, hashlib, random, time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv

# recadrage visage
import numpy as np
import cv2

load_dotenv()

# --- Configuration ---
URLS_CSV   = os.getenv("URLS_CSV", r"C:\Transcript\urls.csv")
RESUME_DIR = Path(os.getenv("RESUME_DIR", r"C:\Transcript\Dropbox (Personal)\resume"))
DRAFTS_DIR = Path(os.getenv("DRAFTS_DIR", "social_drafts")); DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS        = int(os.getenv("MAX_ROWS", "0"))                 # 0 = tout
IMAGE_PROVIDER  = os.getenv("IMAGE_PROVIDER", "hybrid").lower()   # hybrid | openai | unsplash
IMAGE_MODEL     = os.getenv("IMAGE_MODEL", "gpt-image-1")
OPENAI_QUALITY  = os.getenv("OPENAI_QUALITY", "high")             # high|medium|low|auto

# OpenAI client (optionnel)
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

UNSPLASH_BASE = "https://source.unsplash.com"

# ---------------------- Utils ----------------------
def _clean_filename(name: str) -> str:
    name = re.sub(r"[^\w\-\s\.]", "_", name).strip()
    return re.sub(r"\s+", "_", name)[:80]

def _read_resume_for_file(stem: str) -> str:
    """Heuristique: on associe un .txt du dossier RESUME_DIR au stem vidéo."""
    if not RESUME_DIR.exists(): return ""
    cands = list(RESUME_DIR.glob("*.txt"))
    stem_low = stem.lower()
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

def _parse_yt_id(url: str) -> str:
    if not url: return ""
    m = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else ""

def yt_thumb_url(video_id: str) -> str:
    return f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"

def download_with_retry(url: str, tries: int = 3, timeout: int = 20) -> Optional[bytes]:
    if not url:
        return None
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and r.content:
                return r.content
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.6)
    print(f"[WARN] download failed for {url} ({last_err})")
    return None

# -------------- recadrage visage + redressement cover --------------
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
        seed = int(hashlib.md5(str(out_path).encode("utf-8")).hexdigest(), 16)
        rnd  = (seed % 1000) / 1000.0
        bias = max(0.0, min(0.85, focus_top + (rnd - 0.5) * 2 * jitter))
        if im_ratio > tgt_ratio:
            new_h = im.height; new_w = int(new_h * tgt_ratio)
            x0 = (im.width - new_w)//2
            im = im.crop((x0, 0, x0+new_w, new_h))
        else:
            new_w = im.width; new_h = int(new_w / tgt_ratio)
            y0_center = int((im.height - new_h) * bias)
            y0 = max(0, min(im.height-new_h, y0_center))
            im = im.crop((0, y0, new_w, y0+new_h))
        im = im.resize((target_w, target_h), Image.LANCZOS)
    else:
        im = face
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=92)

# ---------------------- Scènes ----------------------
SCENE_BUCKETS = [
    {"key":"interview","match":["interview","figaro","invité","portrait"],
     "prompt_A":"editorial head-and-shoulders portrait, 85mm lens, real estate office background with building posters, natural window light, authentic smile, photorealistic",
     "prompt_B":"editorial half-body portrait, 35mm lens, environmental background (posters/buildings), candid expression, photorealistic",
     "unsplash":"editorial portrait,real estate office,faces,people,window light"},
    {"key":"training","match":["formation","webinar","atelier","présentation","module","apprentissage","cours","coach"],
     "prompt_A":"trainer leading a workshop for real estate agents, wide angle 28mm, large screen, participants engaged, modern classroom, photorealistic",
     "prompt_B":"over-the-shoulder view of attendees taking notes during training, 50mm lens, screen slightly blurred, photorealistic",
     "unsplash":"training workshop,presentation,classroom,business team"},
    {"key":"keys","match":["clé","remise","signature","acte","contrat","vendeur","acheteur"],
     "prompt_A":"close-up of key handover, shallow depth of field, smiling faces in the background, photorealistic",
     "prompt_B":"agent handing a small house keyring to client, mid-shot, natural light, photorealistic",
     "unsplash":"key handover,real estate keys,handshake,contract signing"},
    {"key":"visit","match":["visite","appartement","maison","logement","neuf","ancien","pièces"],
     "prompt_A":"apartment viewing: agent shows a bright modern flat to clients, large windows, 35mm, photorealistic",
     "prompt_B":"clients discussing with agent near window during property visit, 50mm, editorial, photorealistic",
     "unsplash":"apartment interior viewing,real estate showing,wide window,people"},
    {"key":"exterior","match":["bdx","bordeaux","mrs","marseille","quartier","agence","façade","exterieur","extérieur"],
     "prompt_A":"contemporary residential building exterior at golden hour, couple talking with agent, lifestyle, photorealistic",
     "prompt_B":"street-level perspective of modern building facade, people in motion, lifestyle editorial, photorealistic",
     "unsplash":"modern residential building,city street lifestyle,architecture"},
    {"key":"meeting","match":[],
     "prompt_A":"real estate team strategy meeting at a table with documents and tablet, 35mm, photorealistic editorial",
     "prompt_B":"smiling business people in meeting, 50mm, plants and shelves in background, photorealistic",
     "unsplash":"real estate team meeting,collaboration,faces,office"},
]
NEGATIVE = "no illustration, no cartoon, no 3d, no cgi, no render, no watermark, no text"

def pick_bucket(title: str, resume: str) -> Dict:
    text = f"{title} {resume}".lower()
    for b in SCENE_BUCKETS[:-1]:
        if any(k in text for k in b["match"]):
            return b
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest(), 16)
    return SCENE_BUCKETS[h % len(SCENE_BUCKETS)]

# ---------------------- Captions ----------------------
CAPTION_PROMPT = """Tu écris des publications sociales pour un réseau immobilier français.
Génère DEUX variantes pour INSTAGRAM et DEUX variantes pour FACEBOOK à partir du contenu ci-dessous.
Contraintes:
- Ton pro, dynamique, clair.
- Instagram: 150–2200 caractères, 8–12 hashtags pertinents (immobilier, conseils, réseau, local…), pas de lien cliquable.
- Facebook: inclure le lien YouTube fourni (s'il existe) dans le texte (avec UTM source=fb&utm_medium=social&utm_campaign=newsletter), 3–6 hashtags max.
- Retourne en JSON compact avec les clés:
  {"ig":["...varA...","...varB..."], "fb":["...varA...","...varB..."], "title":"titre court punchy"}
Texte de base:
"""

def generate_captions(text: str, title_hint: str, video_url: str) -> dict:
    if not openai_client:
        ig = [f"{title_hint}\n\n{(text or '')[:300]}...\n\n#immobilier #reseau #formation",
              f"Zoom: {title_hint}\n\n{(text or '')[:300]}...\n\n#immobilier #reseau #conseils"]
        fb = [f"{title_hint}\n\n{(text or '')[:300]}...\n\nVoir: {video_url}",
              f"{title_hint} — à découvrir\n\n{(text or '')[:300]}...\n\n{video_url}"]
        return {"ig": ig, "fb": fb, "title": title_hint or "À La Lucarne"}
    payload = CAPTION_PROMPT + f"\nTITRE: {title_hint}\nURL: {video_url}\nTEXTE:\n{text[:2000]}"
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.6,
            messages=[{"role":"user","content": payload}],
        )
        raw = r.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            j = json.loads(m.group(0))
            j["ig"] = [x.strip() for x in j.get("ig", [])][:2]
            j["fb"] = [x.strip() for x in j.get("fb", [])][:2]
            j["title"] = (j.get("title") or title_hint or "À La Lucarne")[:80]
            return j
    except Exception as e:
        print("[WARN] OpenAI captions error:", e)
    return {"ig": [], "fb": [], "title": title_hint or "À La Lucarne"}

# ---------------------- Images (OpenAI/Unsplash) ----------------------
def openai_image(prompt: str, size: str="1536x1024") -> Optional[bytes]:
    if not openai_client:
        return None
    enriched = f"{prompt}, {NEGATIVE}, photorealistic, premium editorial, natural lighting, pleasing color grading"
    sizes = [size, "1024x1024", "1024x1536", "auto"]
    for sz in sizes:
        try:
            im = openai_client.images.generate(
                model=IMAGE_MODEL, prompt=enriched, size=sz, quality=OPENAI_QUALITY, n=1
            )
            b64 = im.data[0].b64_json
            return base64.b64decode(b64)
        except Exception as e:
            print(f"[WARN] OpenAI image failed size={sz}:", e)
    return None

def unsplash_url(keywords: str, size: str, salt: str) -> str:
    h = hashlib.md5((keywords + salt).encode("utf-8")).hexdigest()[:8]
    return f"{UNSPLASH_BASE}/{size}/?{urllib.parse.quote(keywords)}&sig={h}"

def unsplash_bytes_list(keywords: str, size: str, title: str, tag: str, n: int = 3) -> List[Tuple[Optional[bytes], str]]:
    """Retourne jusqu'à n images (bytes, url) depuis Unsplash Source avec seeds variés."""
    items = []
    for i in range(n):
        url = unsplash_url(keywords, size, salt=f"{title}_{tag}_{i}")
        b = download_with_retry(url, tries=3)
        if not b:
            url2 = unsplash_url(keywords + ",portrait", size, salt=f"{title}_{tag}_{i}_alt")
            b = download_with_retry(url2, tries=2)
            if b:
                url = url2
        items.append((b, url))
    return items

def ensure_saved_candidate(img_bytes: Optional[bytes], out_path: Path, w: int, h: int,
                           focus_top: float, jitter: float) -> bool:
    if not img_bytes:
        return False
    try:
        save_resized(img_bytes, out_path, w, h, focus_top, jitter)
        return True
    except Exception:
        return False

# ---------------------- Candidats par vidéo ----------------------
def build_candidates(title: str, resume: str, video_url: str, base_dir: Path) -> Dict:
    """Construit 4 candidats distincts par variante IG/FB."""
    base_dir.mkdir(parents=True, exist_ok=True)
    bA = pick_bucket(title, resume)
    other = [b for b in SCENE_BUCKETS if b["key"] != bA["key"]]
    random.seed(int(hashlib.md5(title.encode("utf-8")).hexdigest(),16))
    bB = random.choice(other) if other else bA

    vid = _parse_yt_id(video_url)
    igA_cands, igB_cands, fbA_cands, fbB_cands = [], [], [], []

    # IG — 1080x1350
    if vid and IMAGE_PROVIDER in {"hybrid"}:
        yt_url = yt_thumb_url(vid)
        b = download_with_retry(yt_url)
        if b and ensure_saved_candidate(b, base_dir/"IG_A_youtube.jpg", 1080, 1350, 0.30, 0.06):
            igA_cands.append({"source":"youtube","image_local":"images/IG_A_youtube.jpg","image_url":yt_url})
    kwA = f"{bA['unsplash']},people,faces,editorial"
    for idx, (b, url) in enumerate(unsplash_bytes_list(kwA, "1080x1350", title, "IG_A", 3), start=1):
        name = f"IG_A_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1080, 1350, 0.30, 0.06):
            igA_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bA["prompt_A"], size="1024x1536")
        if ensure_saved_candidate(b, base_dir/"IG_A_ai.jpg", 1080, 1350, 0.30, 0.06):
            igA_cands.append({"source":"openai","image_local":"images/IG_A_ai.jpg","image_url":""})

    kwB = f"{bB['unsplash']},people,faces,editorial"
    for idx, (b, url) in enumerate(unsplash_bytes_list(kwB, "1080x1350", title, "IG_B", 3), start=1):
        name = f"IG_B_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1080, 1350, 0.28, 0.08):
            igB_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bB["prompt_B"], size="1024x1536")
        if ensure_saved_candidate(b, base_dir/"IG_B_ai.jpg", 1080, 1350, 0.28, 0.08):
            igB_cands.append({"source":"openai","image_local":"images/IG_B_ai.jpg","image_url":""})

    # FB — 1200x630
    kwAfb = f"{bA['unsplash']},people,faces,editorial"
    for idx, (b, url) in enumerate(unsplash_bytes_list(kwAfb, "1600x1066", title, "FB_A", 3), start=1):
        name = f"FB_A_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1200, 630, 0.40, 0.03):
            fbA_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bA["prompt_A"], size="1536x1024")
        if ensure_saved_candidate(b, base_dir/"FB_A_ai.jpg", 1200, 630, 0.40, 0.03):
            fbA_cands.append({"source":"openai","image_local":"images/FB_A_ai.jpg","image_url":""})

    kwBfb = f"{bB['unsplash']},people,faces,editorial"
    for idx, (b, url) in enumerate(unsplash_bytes_list(kwBfb, "1600x1066", title, "FB_B", 3), start=1):
        name = f"FB_B_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1200, 630, 0.46, 0.05):
            fbB_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bB["prompt_B"], size="1536x1024")
        if ensure_saved_candidate(b, base_dir/"FB_B_ai.jpg", 1200, 630, 0.46, 0.05):
            fbB_cands.append({"source":"openai","image_local":"images/FB_B_ai.jpg","image_url":""})

    def pick_selected(cands: List[Dict]) -> int:
        return 0 if cands else -1

    return {
        "ig": {
            "A": {"candidates": igA_cands, "selected": pick_selected(igA_cands)},
            "B": {"candidates": igB_cands, "selected": pick_selected(igB_cands)},
        },
        "fb": {
            "A": {"candidates": fbA_cands, "selected": pick_selected(fbA_cands)},
            "B": {"candidates": fbB_cands, "selected": pick_selected(fbB_cands)},
        },
    }

# -------- brouillon complet ? (meta + captions + ≥1 candidat/variante) --------
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Drafts"):
        fichier = str(row["fichier"])
        stem = Path(fichier).stem
        video_url = str(row.get("url", "")).strip()

        title_hint = re.sub(r"[_\-]+", " ", stem).strip().title()
        resume_text = _read_resume_for_file(stem)

        slug = _clean_filename(stem)
        droot = DRAFTS_DIR / slug
        (droot / "images").mkdir(parents=True, exist_ok=True)

        if _is_complete_draft(droot):
            print(f"[SKIP] {slug} complet déjà généré → ignoré.")
            continue

        # captions
        caps = generate_captions(resume_text, title_hint, video_url)
        ig_caps = (caps.get("ig") or [])[:2]
        fb_caps = (caps.get("fb") or [])[:2]
        title   = caps.get("title") or title_hint

        def _write_if_missing(path: Path, content: str):
            if not path.exists():
                path.write_text(content, encoding="utf-8")
        _write_if_missing(droot / "ig_caption_A.txt", ig_caps[0] if len(ig_caps)>0 else title)
        _write_if_missing(droot / "ig_caption_B.txt", ig_caps[1] if len(ig_caps)>1 else title)
        _write_if_missing(droot / "fb_caption_A.txt", fb_caps[0] if len(fb_caps)>0 else f"{title}\n{video_url}")
        _write_if_missing(droot / "fb_caption_B.txt", fb_caps[1] if len(fb_caps)>1 else f"{title}\n{video_url}")

        # images candidates
        cands = build_candidates(title, resume_text, video_url, droot / "images")

        # meta
        meta_path = droot / "meta.json"
        meta = {
            "video_file": fichier,
            "video_url": video_url,
            "title": title,
            "ig": {
                "A": {"caption_file": "ig_caption_A.txt", **cands["ig"]["A"]},
                "B": {"caption_file": "ig_caption_B.txt", **cands["ig"]["B"]},
            },
            "fb": {
                "A": {"caption_file": "fb_caption_A.txt", **cands["fb"]["A"]},
                "B": {"caption_file": "fb_caption_B.txt", **cands["fb"]["B"]},
            },
            "provider": IMAGE_PROVIDER,
        }
        if meta_path.exists():
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                old = {}
            # On préserve les candidates déjà présentes pour ne pas perdre un tri manuel
            for k1 in ("ig","fb"):
                if k1 in old:
                    for k2 in ("A","B"):
                        if k2 in old[k1] and old[k1][k2].get("candidates"):
                            meta[k1][k2]["candidates"] = old[k1][k2]["candidates"]
                            meta[k1][k2]["selected"]   = old[k1][k2].get("selected", 0)
            meta.update({k:v for k,v in old.items() if k not in meta})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Brouillons créés dans:", DRAFTS_DIR.resolve())

if __name__ == "__main__":
    main()
