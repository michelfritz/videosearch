# generer_posts_sociaux.py — 2 images différentes par variante (IG A/B, FB A/B)
# Provider image paramétrable : hybrid | openai | unsplash
# - hybrid (défaut) : IG = YT/Unsplash (variées, recadrage visage), FB = OpenAI (2 prompts différents)
# - openai : OpenAI pour toutes les images (deux prompts variés par réseau)
# - unsplash : Unsplash pour toutes (requêtes & seeds différents)
# Chemins image_local => toujours "images/....jpg"

import os, re, json, base64, urllib.parse, hashlib, random, time
from pathlib import Path
from typing import Optional, Dict, Tuple
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv

# NEW (recadrage visage)
import numpy as np
import cv2

load_dotenv()

URLS_CSV   = os.getenv("URLS_CSV", r"C:\Transcript\urls.csv")
RESUME_DIR = Path(os.getenv("RESUME_DIR", r"C:\Transcript\Dropbox (Personal)\resume"))
DRAFTS_DIR = Path(os.getenv("DRAFTS_DIR", "social_drafts")); DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

# Vitesse / options
MAX_ROWS        = int(os.getenv("MAX_ROWS", "0"))                 # 0 = tout
IMAGE_PROVIDER  = os.getenv("IMAGE_PROVIDER", "hybrid").lower()   # hybrid | openai | unsplash
IMAGE_MODEL     = os.getenv("IMAGE_MODEL", "gpt-image-1")
OPENAI_QUALITY  = os.getenv("OPENAI_QUALITY", "high")             # high|medium|low|auto

# OpenAI (optionnel)
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and IMAGE_PROVIDER in {"openai","hybrid"}) else None
except Exception:
    openai_client = None

UNSPLASH_BASE = "https://source.unsplash.com"

# ---------------------- Utils ----------------------
def _clean_filename(name: str) -> str:
    name = re.sub(r"[^\w\-\s\.]", "_", name).strip()
    return re.sub(r"\s+", "_", name)[:80]

def _read_resume_for_file(stem: str) -> str:
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

# NEW: téléchargement robuste
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
        time.sleep(0.7)
    print(f"[WARN] download failed for {url} ({last_err})")
    return None

# NEW: crop “face-aware” (repli sur cover si rien détecté)
def _face_aware_crop(im: Image.Image, target_w: int, target_h: int) -> Optional[Image.Image]:
    try:
        gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # plus grand visage
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x + w//2, y + int(h*0.45)  # un peu au-dessus du centre du visage
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
    """Face-aware d’abord, sinon crop cover biais haut + léger jitter."""
    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    face = _face_aware_crop(im, target_w, target_h)
    if face is None:
        im_ratio = im.width / im.height
        tgt_ratio = target_w / target_h
        seed = int(hashlib.md5(str(out_path).encode("utf-8")).hexdigest(), 16)
        rnd  = (seed % 1000) / 1000.0
        bias = max(0.0, min(0.85, focus_top + (rnd - 0.5) * 2 * jitter))
        if im_ratio > tgt_ratio:  # crop largeur
            new_h = im.height
            new_w = int(new_h * tgt_ratio)
            x0 = (im.width - new_w)//2
            im = im.crop((x0, 0, x0+new_w, new_h))
        else:                     # crop hauteur (on remonte vers le haut)
            new_w = im.width
            new_h = int(new_w / tgt_ratio)
            y0_center = int((im.height - new_h) * bias)
            y0 = max(0, min(im.height-new_h, y0_center))
            im = im.crop((0, y0, new_w, y0+new_h))
        im = im.resize((target_w, target_h), Image.LANCZOS)
    else:
        im = face
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=92)

# ---------------------- Buckets scènes ----------------------
SCENE_BUCKETS = [
    {
        "key":"interview",
        "match": ["interview","figaro","invité","portrait"],
        "prompt_A": "editorial head-and-shoulders portrait, 85mm lens, real estate office background with building posters, natural window light, authentic smile, photorealistic",
        "prompt_B": "editorial half-body portrait, 35mm lens, environmental background (posters/buildings), candid expression, photorealistic",
        "unsplash": "editorial portrait,real estate office,faces,people,window light"
    },
    {
        "key":"training",
        "match": ["formation","webinar","atelier","présentation","module","apprentissage","cours","coach"],
        "prompt_A": "trainer leading a workshop for real estate agents, wide angle 28mm, large screen, participants engaged, modern classroom, photorealistic",
        "prompt_B": "over-the-shoulder view of attendees taking notes during training, 50mm lens, screen slightly blurred, photorealistic",
        "unsplash": "training workshop,presentation,classroom,business team"
    },
    {
        "key":"keys",
        "match": ["clé","remise","signature","acte","contrat","vendeur","acheteur"],
        "prompt_A": "close-up of key handover, shallow depth of field, smiling faces in the background, photorealistic",
        "prompt_B": "agent handing a small house keyring to client, mid-shot, natural light, photorealistic",
        "unsplash": "key handover,real estate keys,handshake,contract signing"
    },
    {
        "key":"visit",
        "match": ["visite","appartement","maison","logement","neuf","ancien","pièces"],
        "prompt_A": "apartment viewing: agent shows a bright modern flat to clients, large windows, 35mm, photorealistic",
        "prompt_B": "clients discussing with agent near window during property visit, 50mm, editorial, photorealistic",
        "unsplash": "apartment interior viewing,real estate showing,wide window,people"
    },
    {
        "key":"exterior",
        "match": ["bdx","bordeaux","mrs","marseille","quartier","agence","façade","exterieur","extérieur"],
        "prompt_A": "contemporary residential building exterior at golden hour, couple talking with agent, lifestyle, photorealistic",
        "prompt_B": "street-level perspective of modern building facade, people in motion, lifestyle editorial, photorealistic",
        "unsplash": "modern residential building,city street lifestyle,architecture"
    },
    {
        "key":"meeting",
        "match": [],
        "prompt_A": "real estate team strategy meeting at a table with documents and tablet, 35mm, photorealistic editorial",
        "prompt_B": "smiling business people in meeting, 50mm, plants and shelves in background, photorealistic",
        "unsplash": "real estate team meeting,collaboration,faces,office"
    },
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
            model="gpt-4o-mini",
            temperature=0.6,
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
                model=IMAGE_MODEL,
                prompt=enriched,
                size=sz,
                quality=OPENAI_QUALITY,
                n=1,
            )
            b64 = im.data[0].b64_json
            return base64.b64decode(b64)
        except Exception as e:
            print(f"[WARN] OpenAI image failed size={sz}:", e)
    return None

def unsplash_url(keywords: str, size: str, salt: str) -> str:
    h = hashlib.md5((keywords + salt).encode("utf-8")).hexdigest()[:8]
    return f"{UNSPLASH_BASE}/{size}/?{urllib.parse.quote(keywords)}&sig={h}"

# PATCH: Unsplash robuste (variantes + retry)
def two_unsplash_bytes(kwA: str, kwB: str, sizeA: str, sizeB: str, title: str) -> Tuple[Optional[bytes], Optional[bytes], str, str]:
    """
    Essaie Unsplash Source avec seed. Si 503, on réessaie avec variantes (ajout 'portrait').
    Retourne (bytesA, bytesB, urlA, urlB).
    """
    urlA = unsplash_url(kwA, sizeA, salt=title+"_A")
    urlB = unsplash_url(kwB, sizeB, salt=title+"_B")

    bA = download_with_retry(urlA, tries=3)
    bB = download_with_retry(urlB, tries=3)

    if not bA:
        altA = unsplash_url(kwA + ",portrait", sizeA, salt=title+"_A2")
        bA2 = download_with_retry(altA, tries=2)
        if bA2:
            urlA = altA
            bA = bA2

    if not bB:
        altB = unsplash_url(kwB + ",portrait", sizeB, salt=title+"_B2")
        bB2 = download_with_retry(altB, tries=2)
        if bB2:
            urlB = altB
            bB = bB2

    return bA, bB, urlA, urlB

def two_openai_bytes(promptA: str, promptB: str, sizeA: str, sizeB: str) -> Tuple[Optional[bytes], Optional[bytes]]:
    return openai_image(promptA, size=sizeA), openai_image(promptB, size=sizeB)

# ---------------------- Construction images par vidéo ----------------------
def build_images_for_video(title: str, resume: str, video_url: str, base_dir: Path) -> Dict:
    """
    Produit 4 visuels distincts :
      IG_A, IG_B, FB_A, FB_B   (toujours différents)
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    bA = pick_bucket(title, resume)
    other = [b for b in SCENE_BUCKETS if b["key"] != bA["key"]]
    random.seed(int(hashlib.md5(title.encode("utf-8")).hexdigest(),16))
    bB = random.choice(other) if other else bA

    # --- IG (1080x1350) : selon provider
    vid = _parse_yt_id(video_url)
    igA_public = yt_thumb_url(vid) if vid and IMAGE_PROVIDER in {"hybrid"} else None

    if IMAGE_PROVIDER == "openai":
        igA_bytes, igB_bytes = two_openai_bytes(bA["prompt_A"], bB["prompt_B"], "1024x1536", "1024x1536")
        igA_public = ""  # pas d’URL publique
        igB_public = ""
    elif IMAGE_PROVIDER == "unsplash":
        kwA = f"{bA['unsplash']},people,faces,editorial"
        kwB = f"{bB['unsplash']},people,faces,editorial"
        igA_bytes, igB_bytes, urlA, urlB = two_unsplash_bytes(kwA, kwB, "1080x1350", "1080x1350", title)
        igA_public = urlA
        igB_public = urlB
        # Fallback OpenAI si Unsplash 503
        if (igA_bytes is None) and openai_client:
            igA_bytes = openai_image(bA["prompt_A"], size="1024x1536")
            igA_public = ""
        if (igB_bytes is None) and openai_client:
            igB_bytes = openai_image(bB["prompt_B"], size="1024x1536")
            igB_public = ""
    else:  # hybrid
        # IG_A : YT si dispo sinon Unsplash bucket A
        if not igA_public:
            urlA = unsplash_url(f"{bA['unsplash']},people,faces,editorial", "1080x1350", title+"_IGA")
            igA_public = urlA
        igA_bytes = download_with_retry(igA_public)
        # fallback si miniature YT KO
        if not igA_bytes:
            urlA = unsplash_url(f"{bA['unsplash']},people,faces,editorial", "1080x1350", title+"_IGA_fallback")
            igA_public = urlA
            igA_bytes  = download_with_retry(igA_public)
        if (igA_bytes is None) and openai_client:
            igA_bytes = openai_image(bA["prompt_A"], size="1024x1536")
            igA_public = ""

        # IG_B : Unsplash bucket B
        urlB = unsplash_url(f"{bB['unsplash']},people,faces,editorial", "1080x1350", title+"_IGB")
        igB_public = urlB
        igB_bytes  = download_with_retry(igB_public)
        if not igB_bytes:
            urlB = unsplash_url(f"{bB['unsplash']},people,faces,editorial", "1080x1350", title+"_IGB_fallback")
            igB_public = urlB
            igB_bytes  = download_with_retry(igB_public)
        if (igB_bytes is None) and openai_client:
            igB_bytes = openai_image(bB["prompt_B"], size="1024x1536")
            igB_public = ""

    igA_local = base_dir / "IG_A.jpg"
    igB_local = base_dir / "IG_B.jpg"
    if igA_bytes: save_resized(igA_bytes, igA_local, 1080, 1350, focus_top=0.30, jitter=0.06)
    if igB_bytes: save_resized(igB_bytes, igB_local, 1080, 1350, focus_top=0.28, jitter=0.08)

    # --- FB (1200x630) : selon provider
    if IMAGE_PROVIDER == "openai":
        fbA_bytes, fbB_bytes = two_openai_bytes(bA["prompt_A"], bB["prompt_B"], "1536x1024", "1536x1024")
    elif IMAGE_PROVIDER == "unsplash":
        kwA = f"{bA['unsplash']},people,faces,editorial"
        kwB = f"{bB['unsplash']},people,faces,editorial"
        fbA_bytes, fbB_bytes, uA, uB = two_unsplash_bytes(kwA, kwB, "1600x1066", "1600x1066", title+"_FB")
        # Fallback OpenAI si Unsplash 503
        if (fbA_bytes is None) and openai_client:
            fbA_bytes = openai_image(bA["prompt_A"], size="1536x1024")
        if (fbB_bytes is None) and openai_client:
            fbB_bytes = openai_image(bB["prompt_B"], size="1536x1024")
    else:  # hybrid: FB via OpenAI (deux prompts variés)
        fbA_bytes, fbB_bytes = two_openai_bytes(bA["prompt_A"], bB["prompt_B"], "1536x1024", "1536x1024")
        # fallback Unsplash si OpenAI indispo
        if not fbA_bytes or not fbB_bytes:
            kwA = f"{bA['unsplash']},people,faces,editorial"
            kwB = f"{bB['unsplash']},people,faces,editorial"
            fbA_bytes2, fbB_bytes2, _, _ = two_unsplash_bytes(kwA, kwB, "1600x1066", "1600x1066", title+"_FBf")
            fbA_bytes = fbA_bytes or fbA_bytes2
            fbB_bytes = fbB_bytes or fbB_bytes2

    fbA_local = base_dir / "FB_A.jpg"
    fbB_local = base_dir / "FB_B.jpg"
    if fbA_bytes: save_resized(fbA_bytes, fbA_local, 1200, 630, focus_top=0.40, jitter=0.03)
    if fbB_bytes: save_resized(fbB_bytes, fbB_local, 1200, 630, focus_top=0.05, jitter=0.05)

    results = {"ig": {"A":{}, "B":{}}, "fb": {"A":{}, "B":{}}}
    results["ig"]["A"] = {"image_local": f"images/{igA_local.name}", "image_url": igA_public or ""}
    results["ig"]["B"] = {"image_local": f"images/{igB_local.name}", "image_url": igB_public or ""}
    if fbA_local.exists(): results["fb"]["A"] = {"image_local": f"images/{fbA_local.name}"}
    if fbB_local.exists(): results["fb"]["B"] = {"image_local": f"images/{fbB_local.name}"}
    results["bucketA"] = bA["key"]; results["bucketB"] = bB["key"]
    return results

# ---------------------- Captions + SKIP si déjà généré ----------------------
def _is_complete_draft(root: Path) -> bool:
    required = [
        "meta.json",
        "ig_caption_A.txt", "ig_caption_B.txt",
        "fb_caption_A.txt", "fb_caption_B.txt",
        "images/IG_A.jpg", "images/IG_B.jpg",
        "images/FB_A.jpg", "images/FB_B.jpg",
    ]
    return all((root / p).exists() for p in required)

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

        # --- SKIP si brouillon complet déjà présent
        if _is_complete_draft(droot):
            print(f"[SKIP] {slug} complet déjà généré → ignoré.")
            continue

        (droot / "images").mkdir(parents=True, exist_ok=True)

        # Génération captions
        caps = generate_captions(resume_text, title_hint, video_url)
        ig_caps = (caps.get("ig") or [])[:2]
        fb_caps = (caps.get("fb") or [])[:2]
        title   = caps.get("title") or title_hint

        imgs = build_images_for_video(title, resume_text, video_url, droot / "images")

        # Écritures (on réécrit si absent seulement)
        def _write_if_missing(path: Path, content: str):
            if not path.exists():
                path.write_text(content, encoding="utf-8")

        _write_if_missing(droot / "ig_caption_A.txt", ig_caps[0] if len(ig_caps)>0 else title)
        _write_if_missing(droot / "ig_caption_B.txt", ig_caps[1] if len(ig_caps)>1 else title)
        _write_if_missing(droot / "fb_caption_A.txt", fb_caps[0] if len(fb_caps)>0 else f"{title}\n{video_url}")
        _write_if_missing(droot / "fb_caption_B.txt", fb_caps[1] if len(fb_caps)>1 else f"{title}\n{video_url}")

        meta = {
            "video_file": fichier,
            "video_url": video_url,
            "title": title,
            "ig": {
                "A": {"caption_file": "ig_caption_A.txt", **imgs["ig"]["A"]},
                "B": {"caption_file": "ig_caption_B.txt", **imgs["ig"]["B"]},
            },
            "fb": {
                "A": {"caption_file": "fb_caption_A.txt", **imgs["fb"].get("A", {})},
                "B": {"caption_file": "fb_caption_B.txt", **imgs["fb"].get("B", {})},
            },
            "buckets": {"A": imgs.get("bucketA"), "B": imgs.get("bucketB")},
            "provider": IMAGE_PROVIDER,
        }
        # n’écrase pas si présent (mais met à jour si incomplet)
        meta_path = droot / "meta.json"
        if meta_path.exists():
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                old = {}
            old.update(meta)  # merge simple
            meta = old
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Brouillons créés dans:", DRAFTS_DIR.resolve())

if __name__ == "__main__":
    main()

