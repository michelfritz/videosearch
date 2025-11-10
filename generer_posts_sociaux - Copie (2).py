# generer_posts_sociaux.py — Brouillons “flex” + incrémentalité
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

# --- Incrémentalité ---
# SKIP_IF_COMPLETE=1 : si un brouillon complet (meta + 4x captions + ≥1 image par variante) existe, on saute TOUT
SKIP_IF_COMPLETE = os.getenv("SKIP_IF_COMPLETE", "1") == "1"

# --- Configuration ---
URLS_CSV   = os.getenv("URLS_CSV", r"C:\Transcript\urls.csv")
RESUME_DIR = Path(os.getenv("RESUME_DIR", r"C:\Transcript\Dropbox (Personal)\resume"))
DRAFTS_DIR = Path(os.getenv("DRAFTS_DIR", "social_drafts")); DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS        = int(os.getenv("MAX_ROWS", "0"))                 # 0 = tout
IMAGE_PROVIDER  = os.getenv("IMAGE_PROVIDER", "hybrid").lower()   # hybrid | openai | unsplash
IMAGE_MODEL     = os.getenv("IMAGE_MODEL", "gpt-image-1")
OPENAI_QUALITY  = os.getenv("OPENAI_QUALITY", "high")             # high|medium|low|auto

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
    """Heuristique: on associe un .txt du dossier RESUME_DIR au stem vidéo. (Améliorée)"""
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
    """Titre sans mentions internes (visio, webinar…)"""
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
    if not url:
        return None
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers or {})
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

# ---------------------- Scènes & thématiques (inchangé) ----------------------
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

# ====================== Reste du pipeline (inchangé) ======================
_FRENCH_CITIES = [
    "Paris","Lyon","Marseille","Bordeaux","Toulouse","Nice","Nantes","Montpellier","Rennes","Strasbourg",
    "Lille","Grenoble","Rouen","Reims","Toulon","Saint-Étienne","Dijon","Angers","Le Havre","Clermont-Ferrand",
    "Brest","Nancy","Tours","Aix-en-Provence","Annecy","Bayonne","Pau","La Rochelle","Metz","Orléans","Perpignan"
]
_CITY_REGEX = re.compile(r"\b(" + "|".join([re.escape(c) for c in _FRENCH_CITIES]) + r")\b", re.I)

TOPIC_RULES = [
    {
        "key": "taux_credit",
        "label": "Taux & crédit immobilier",
        "match": [r"\btaux", r"cr[ée]dit", r"emprunt", r"banque", r"mensualit[ée]s?", r"assurance emprunteur", r"oat", r"usure"],
        "unsplash": "mortgage papers,signing documents,calculator,home loan,real estate office",
        "unsplash_extra_people": True,
        "prompt_A": "clients signing mortgage paperwork with agent, calculator and documents on a table, modern real estate office, natural light, photorealistic",
        "prompt_B": "close-up hands signing home loan with calculator and french paperwork, shallow depth of field, photorealistic",
        "hashtags": ["#taux", "#creditimmobilier", "#financement", "#achatimmobilier", "#immobilier", "#banque", "#emprunt"]
    },
    {
        "key": "dpe_renovation",
        "label": "DPE & rénovation énergétique",
        "match": [r"\bdpe\b", r"r[ée]novation", r"isolati", r"pompe [àa] chaleur", r"r[ée]glementation [ée]nerg[ée]tique", r"passoire", r"thermique"],
        "unsplash": "home insulation,energy efficiency,heat pump,solar panels,thermographic,eco renovation",
        "unsplash_extra_people": False,
        "prompt_A": "technician inspecting home insulation and heat pump installation, modern house, photorealistic",
        "prompt_B": "homeowner reviewing energy renovation plan with contractor, documents and tablet, photorealistic",
        "hashtags": ["#DPE", "#rénovation", "#énergie", "#logement", "#transition", "#immobilier", "#travaux"]
    },
    {
        "key": "fiscalite_pinel_ptz",
        "label": "Fiscalité (Pinel, PTZ, LMNP…) ",
        "match": [r"\bpinel\+?", r"\bptz\b", r"\blmnp\b", r"\blmp\b", r"d[ée]ficit foncier", r"amortissement", r"ifi", r"taxe fonci[èe]re"],
        "unsplash": "new residential building,blueprints with calculator,real estate contract,construction crane",
        "unsplash_extra_people": False,
        "prompt_A": "couple reviewing tax incentive real estate investment with advisor in office, documents and laptop, photorealistic",
        "prompt_B": "exterior of new residential building at golden hour, lifestyle editorial, photorealistic",
        "hashtags": ["#Pinel", "#PTZ", "#LMNP", "#fiscalité", "#investissement", "#immobilier", "#neuf"]
    },
    {
        "key": "prix_transactions",
        "label": "Prix au m² & volumes de transactions",
        "match": [r"prix", r"€/m", r"m²", r"transactions?", r"barom[èe]tre", r"notaires", r"indice", r"hausse|baisse|stagnation"],
        "unsplash": "business meeting with laptop charts,real estate analytics,market charts on screen",
        "unsplash_extra_people": True,
        "prompt_A": "real estate analyst showing price per square meter charts on laptop to clients, photorealistic",
        "prompt_B": "close-up of market charts and notebook on desk in meeting, photorealistic",
        "hashtags": ["#priximmobilier", "#marchéimmobilier", "#transactions", "#notaires", "#baromètre", "#m2", "#tendance"]
    },
    {
        "key": "location_bail",
        "label": "Location & bail (encadrement, IRL, loyers)",
        "match": [r"loyer", r"bail", r"encadrement", r"\birl\b", r"locati", r"gli", r"mobilit[ée]"],
        "unsplash": "apartment interior for rent,for rent sign,tenant meeting agent,lease signing",
        "unsplash_extra_people": True,
        "prompt_A": "agent presenting rental apartment to tenants, bright interior, photorealistic",
        "prompt_B": "close-up of lease signing with keys on table, photorealistic",
        "hashtags": ["#location", "#bail", "#loyer", "#IRL", "#locatif", "#immobilier", "#conseils"]
    },
    {
        "key": "copropriete",
        "label": "Copropriété, charges & AG",
        "match": [r"copropri[ée]t[ée]", r"\bsyndic\b", r"\bAG\b", r"charges", r"fonds de travaux", r"ALUR|ELAN"],
        "unsplash": "apartment building facade balconies,hoa meeting,condominium exterior",
        "unsplash_extra_people": False,
        "prompt_A": "apartment building facade with balconies, daylight, photorealistic",
        "prompt_B": "homeowners association meeting with documents on table, photorealistic",
        "hashtags": ["#copropriété", "#charges", "#AG", "#syndic", "#immobilier", "#résidentiel"]
    },
    {
        "key": "neuf_construction",
        "label": "Neuf & construction (VEFA, RE2020)",
        "match": [r"\bvefa\b", r"\bneuf\b", r"promoteur", r"re2020|rt2012", r"grue|chantier|permis de construire"],
        "unsplash": "construction site crane,new apartments,architectural blueprint",
        "unsplash_extra_people": False,
        "prompt_A": "new residential building under construction with cranes, photorealistic",
        "prompt_B": "architect and client reviewing blueprints at construction site, photorealistic",
        "hashtags": ["#immobilierneuf", "#VEFA", "#RE2020", "#construction", "#promoteur"]
    },
    {
        "key": "investissement",
        "label": "Investissement locatif & rendement",
        "match": [r"rentabilit[ée]", r"cash ?flow", r"colocation", r"saisonni[èe]re|airbnb", r"vacance locative"],
        "unsplash": "investor analyzing returns on laptop,apartment interior,real estate investment meeting",
        "unsplash_extra_people": True,
        "prompt_A": "investor reviewing rental ROI with advisor on laptop, apartment background, photorealistic",
        "prompt_B": "stylish rental apartment interior prepared for tenants, photorealistic",
        "hashtags": ["#investissement", "#locatif", "#rentabilité", "#immobilier", "#cashflow"]
    },
    {
        "key": "mandat_transaction",
        "label": "Mandats, compromis, signatures",
        "match": [r"mandat", r"exclusif", r"compromis", r"promesse", r"offre d'achat", r"notaire", r"signature", r"acte authentique"],
        "unsplash": "handshake with keys,contract signing,real estate deal",
        "unsplash_extra_people": True,
        "prompt_A": "agent handing keys to buyer after signing, smiles, photorealistic",
        "prompt_B": "close-up of signing real estate contract with pen and keys on table, photorealistic",
        "hashtags": ["#mandat", "#signature", "#compromis", "#notaire", "#achatimmobilier"]
    },
    {
        "key": "legal_reglement",
        "label": "Légal & réglementation (lois, décrets)",
        "match": [r"\bloi\b", r"d[ée]cret", r"arr[êe]t[ée]", r"obligation", r"diagnostic", r"plafonds?"],
        "unsplash": "legal documents on desk,advisor meeting,official paperwork",
        "unsplash_extra_people": True,
        "prompt_A": "advisor explaining new housing law to clients with printed documents, photorealistic",
        "prompt_B": "legal documents and pen on wooden desk, shallow depth of field, photorealistic",
        "hashtags": ["#réglementation", "#loi", "#diagnostics", "#immobilier"]
    },
]

AUDIENCES = ("acheteurs","vendeurs","candidats")
_CANDIDATE_PATTERNS = [
    r"recrut", r"deven(?:ez|ir)\s+agent", r"nous rejoindre", r"mandataire",
    r"commission", r"statut\s+ind[ée]pendant", r"candidats?", r"postule[rz]?"
]

def pick_audiences(text: str, title: str) -> Tuple[str, str]:
    base = (title or "") + "\n" + (text or "")
    score_cand = sum(len(re.findall(p, base, flags=re.I)) for p in _CANDIDATE_PATTERNS)
    if score_cand >= 2:
        return ("acheteurs","candidats")
    return ("acheteurs","vendeurs")

def _sanitize_b2c_text(s: str) -> str:
    if not s: return s
    s = re.sub(r"https?://\S+", "", s)
    banned = r"(webinar|webinaire|visi[oô]|formation\s+interne|atelier|module|newsletter|youtube|vid[ée]o)"
    lines = [ln for ln in s.splitlines() if not re.search(banned, ln, flags=re.I)]
    s = "\n".join(lines)
    s = re.sub(r"\bagents?\s+immobiliers?\b", "conseillers immobiliers", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _cta_for_persona(persona: str, city: Optional[str]) -> str:
    if persona == "acheteurs":
        return "Besoin d’y voir clair ? Parlons de votre projet en message privé."
    if persona == "vendeurs":
        prefix = f"{city}: " if city else ""
        return f"{prefix}Estimation offerte et stratégie sur-mesure. Écrivez-nous."
    return "Envie de devenir conseiller immobilier ? Échangeons en message privé."

def _norm(s: str) -> str:
    return (s or "").lower()

def _find_cities(text: str, title: str) -> List[str]:
    found = set()
    for m in _CITY_REGEX.finditer((title or "") + " " + (text or "")):
        found.add(m.group(1).title())
    return list(found)[:3]

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

def _topic_to_bucket(topic_rule: dict, cities: List[str]) -> Dict:
    rule = topic_rule["rule"]
    pA = rule["prompt_A"] + (f", city of {cities[0]}" if cities and rule["key"] in {"prix_transactions","marche_local","neuf_construction"} else "")
    pB = rule["prompt_B"] + (f", city of {cities[0]}" if cities and rule["key"] in {"prix_transactions","marche_local","neuf_construction"} else "")
    return {
        "key": rule["key"],
        "match": [rule["label"]],
        "prompt_A": pA,
        "prompt_B": pB,
        "unsplash": (rule["unsplash"] + (f",{cities[0]} architecture" if cities and rule["key"] in {"prix_transactions","neuf_construction","copropriete"} else "")),
        "unsplash_extra": ("people,faces,editorial" if rule.get("unsplash_extra_people") else "architecture,exterior,building,home interior")
    }

def _fallback_pick_bucket(title: str, resume: str) -> Dict:
    text = f"{title} {resume}".lower()
    for b in SCENE_BUCKETS[:-1]:
        if any(k in text for k in b["match"]):
            return {**b, "unsplash_extra": "people,faces,editorial"}
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest(), 16)
    b = SCENE_BUCKETS[h % len(SCENE_BUCKETS)]
    return {**b, "unsplash_extra": "people,faces,editorial"}

def choose_buckets(title: str, resume: str) -> Tuple[Dict, Dict]:
    topics = _extract_topics(resume, title)
    cities = _find_cities(resume, title)
    if topics:
        bA = _topic_to_bucket(topics[0], cities)
        if len(topics) > 1:
            bB = _topic_to_bucket(topics[1], cities)
        else:
            bB = dict(bA)
            bB["prompt_A"], bB["prompt_B"] = bA["prompt_B"], bA["prompt_A"]
        return bA, bB
    bA = _fallback_pick_bucket(title, resume)
    other = [b for b in SCENE_BUCKETS if b["key"] != bA["key"]]
    random.seed(int(hashlib.md5(title.encode("utf-8")).hexdigest(),16))
    b0 = random.choice(other) if other else bA
    bB = {**b0, "unsplash_extra": "people,faces,editorial"}
    return bA, bB

CAPTION_PROMPT = """Tu écris des publications sociales B2C pour un public français (acheteurs / vendeurs ; optionnellement candidats si le texte parle clairement de recrutement).
Objectif: t'appuyer STRICTEMENT sur le texte fourni (résumé) et les Faits clés, sans inventer.
Produis DEUX variantes pour INSTAGRAM et DEUX variantes pour FACEBOOK.
Règles:
- Cible principale: A = acheteurs ; B = vendeurs. (Si le texte parle clairement de recrutement, tu peux orienter une variante "candidats".)
- NE t'adresse PAS aux "agents immobiliers" ni au "réseau interne". Pas de mention de visio, webinar, formation interne.
- N'inclus AUCUN lien (pas de YouTube / pas de vidéo). Pas d'appel du type "voir la vidéo".
- Ton pro, dynamique, clair, orienté bénéfices clients. Intègre des chiffres/dates/% uniquement s'ils figurent dans le texte.
- Instagram: 150–2200 caractères, 8–12 hashtags pertinents par variante (immobilier, crédit, DPE, marché, ville…), pas de lien cliquable.
- Facebook: 3–6 hashtags, pas de lien.
- Termine chaque variante par un court CTA adapté (ex: estimation offerte, parler du projet, message privé).
- Retourne en JSON compact:
  {{\"ig\":[\"...varA...\",\"...varB...\"], \"fb\":[\"...varA...\",\"...varB...\"], \"title\":\"titre court punchy\"}}
ENTRÉES:
- TITRE: {title_hint}
- SUJETS: {topics_inline}
- FAITS: {facts_inline}
TEXTE:
{text}
"""

def _extract_fact_snippets(text: str, max_items: int = 6) -> List[str]:
    if not text: return []
    parts = re.split(r"[\\.\n\\r;]+", text)
    facts = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if re.search(r"\\b20\\d{2}\\b|€|%|points? de base|taux|prix|m²|\\bIRL\\b|\\bDPE\\b|\\bPTZ\\b|\\bPinel\\+?\\b|\\bLMNP\\b|\\bOAT\\b|transactions?", p, re.I):
            facts.append(p)
    seen = set(); out = []
    for f in facts:
        f2 = f[:220].strip()
        if f2.lower() not in seen:
            out.append(f2); seen.add(f2.lower())
        if len(out) >= max_items: break
    return out

def _topic_hashtags(key: str, extra_city: Optional[str]=None, persona: Optional[str]=None, n_ig:int=10, n_fb:int=5) -> Tuple[List[str], List[str]]:
    base = ["#immobilier", "#conseilsimmobilier"]
    for r in TOPIC_RULES:
        if r["key"] == key:
            base = list(dict.fromkeys(r["hashtags"] + ["#immobilier"]))
            break
    if extra_city:
        base = base + [f"#{extra_city.replace(' ', '')}", "#marchélocal"]
    persona_tags: List[str] = []
    if persona == "acheteurs":
        persona_tags = ["#acheteurs", "#projetimmobilier", "#premierachat"]
    elif persona == "vendeurs":
        persona_tags = ["#vendeurs", "#estimationgratuite", "#vendre"]
    elif persona == "candidats":
        persona_tags = ["#recrutement", "#devenezconseiller", "#carrière"]
    base = list(dict.fromkeys(base + persona_tags))
    ig = (base + ["#réseau", "#actualité", "#france"])[:max(8, min(n_ig, 12))]
    fb = (base + ["#actualité"])[:max(3, min(n_fb, 6))]
    return ig, fb

def _compose_caption_variant(topic_key: str, title_hint: str, facts: List[str], city: Optional[str], platform: str, persona: str) -> str:
    ig_hash, fb_hash = _topic_hashtags(topic_key, city, persona)
    facts_txt = ""
    if facts:
        facts_txt = " • ".join(facts[:3])
    safe_title = _public_title(title_hint)
    if platform == "ig":
        label = next((r["label"] for r in TOPIC_RULES if r["key"] == topic_key), safe_title)
        hook = f"Zoom sur {label.lower()}" if label else "Zoom sur l’actualité"
        body = f"{hook} — {safe_title}\n\n{facts_txt}\n\n{_cta_for_persona(persona, city)}\n\n"
        txt = _sanitize_b2c_text(body + " ".join(ig_hash))
    else:
        label = next((r["label"] for r in TOPIC_RULES if r["key"] == topic_key), "")
        intro = f"{label} — " if label else ""
        body = f"{intro}{safe_title}\n\n{facts_txt}\n\n{_cta_for_persona(persona, city)}\n\n"
        txt = _sanitize_b2c_text(body + " ".join(fb_hash))
    if len(txt.strip()) < 60:
        mini = f"{_sanitize_b2c_text(safe_title)} — {_cta_for_persona(persona, city)}\n\n" + " ".join((ig_hash if platform=='ig' else fb_hash))
        return mini.strip()
    return txt.strip()

def _local_caption_pack(text: str, title_hint: str) -> dict:
    topics = _extract_topics(text, title_hint)
    topic_keys = [t["key"] for t in topics[:2]] or ["mandat_transaction", "prix_transactions"]
    facts = _extract_fact_snippets(text)
    cities = _find_cities(text, title_hint)
    city_for_hash = cities[0] if cities else None
    audA, audB = pick_audiences(text, title_hint)
    tA = topic_keys[0]
    tB = topic_keys[1] if len(topic_keys) > 1 else topic_keys[0]
    ig = [
        _compose_caption_variant(tA, title_hint, facts, city_for_hash, "ig", audA),
        _compose_caption_variant(tB, "Le point à retenir", facts[2:]+facts[:2], city_for_hash, "ig", audB),
    ]
    fb = [
        _compose_caption_variant(tA, title_hint, facts, city_for_hash, "fb", audA),
        _compose_caption_variant(tB, "À savoir cette semaine", facts[2:]+facts[:2], city_for_hash, "fb", audB),
    ]
    return {"ig": ig, "fb": fb, "title": (_public_title(title_hint) or "À La Lucarne")[:80]}

def generate_captions(text: str, title_hint: str, video_url: str) -> dict:
    if openai_client:
        topics = _extract_topics(text, title_hint)
        facts = _extract_fact_snippets(text)
        topics_inline = ", ".join([t["rule"]["label"] for t in topics[:4]]) or "Actualité immobilière"
        facts_inline = " | ".join(facts) if facts else "—"
        payload = CAPTION_PROMPT.format(
            title_hint=_public_title(title_hint), topics_inline=topics_inline, facts_inline=facts_inline, text=text[:4000]
        )
        try:
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.6,
                messages=[{"role":"user","content": payload}],
            )
            raw = r.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                import json as _json
                j = _json.loads(m.group(0))
                ig = [_sanitize_b2c_text((x or "").strip()) for x in j.get("ig", []) if (x or "").strip()]
                fb = [_sanitize_b2c_text((x or "").strip()) for x in j.get("fb", []) if (x or "").strip()]
                title = (j.get("title") or _public_title(title_hint) or "À La Lucarne")[:80]
                if len(ig) < 2 or len(fb) < 2:
                    local = _local_caption_pack(text, title_hint)
                    ig = (ig + local["ig"])[:2]
                    fb = (fb + local["fb"])[:2]
                ig = [s if len(s.strip()) >= 40 else _local_caption_pack(text, title_hint)["ig"][i] for i, s in enumerate(ig)]
                fb = [s if len(s.strip()) >= 40 else _local_caption_pack(text, title_hint)["fb"][i] for i, s in enumerate(fb)]
                return {"ig": ig, "fb": fb, "title": title}
        except Exception as e:
            print("[WARN] OpenAI captions error:", e)
    return _local_caption_pack(text, title_hint)

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
            import base64 as _b64
            return _b64.b64decode(b64)
        except Exception as e:
            print(f"[WARN] OpenAI image failed size={sz}:", e)
    return None

def _unsplash_headers() -> dict:
    if not UNSPLASH_ACCESS_KEY:
        raise RuntimeError("UNSPLASH_ACCESS_KEY manquant dans l'environnement (.env).")
    return {
        "Accept-Version": "v1",
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
    }

def _unsplash_search(query: str, orientation: Optional[str]=None, per_page: int=30) -> List[dict]:
    params = {"query": query, "per_page": max(1, min(per_page, 30)), "content_filter": "high"}
    if orientation in {"portrait","landscape","squarish"}:
        params["orientation"] = orientation
    try:
        r = requests.get(f"{UNSPLASH_API_BASE}/search/photos", headers=_unsplash_headers(), params=params, timeout=20)
        if r.status_code == 200:
            data = r.json() or {}
            return data.get("results", []) or []
        print("[WARN] Unsplash search status:", r.status_code, r.text[:200])
    except Exception as e:
        print("[WARN] Unsplash search error:", e)
    return []

def _unsplash_random(query: str, orientation: Optional[str]=None, count: int=3) -> List[dict]:
    params = {"query": query, "count": max(1, min(count, 30)), "content_filter": "high"}
    if orientation in {"portrait","landscape","squarish"}:
        params["orientation"] = orientation
    try:
        r = requests.get(f"{UNSPLASH_API_BASE}/photos/random", headers=_unsplash_headers(), params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list): return data
            elif isinstance(data, dict): return [data]
        print("[WARN] Unsplash random status:", r.status_code, r.text[:200])
    except Exception as e:
        print("[WARN] Unsplash random error:", e)
    return []

def _pick_deterministic(items: List[dict], title: str, tag: str, i: int) -> Optional[dict]:
    if not items: return None
    idx = int(hashlib.md5(f"{title}|{tag}|{i}".encode("utf-8")).hexdigest()[:8], 16) % len(items)
    return items[idx]

def unsplash_bytes_list(keywords: str, desired_size: str, title: str, tag: str, n: int = 3, orientation: Optional[str]=None) -> List[Tuple[Optional[bytes], str, dict]]:
    items = []
    results = _unsplash_search(keywords, orientation=orientation, per_page=UNSPLASH_PER_PAGE)
    if not results:
        results = _unsplash_random(keywords, orientation=orientation, count=max(3, n))
    for i in range(n):
        picked = _pick_deterministic(results, title, tag, i)
        if not picked:
            items.append((None, "", {}))
            continue
        url_dl = picked.get("urls", {}).get("regular") or picked.get("urls", {}).get("full") or picked.get("urls", {}).get("raw") or ""
        img_bytes = download_with_retry(url_dl, tries=3)
        photographer = (picked.get("user") or {}).get("name") or ""
        profile_url  = ((picked.get("user") or {}).get("links") or {}).get("html") or ""
        photo_url    = (picked.get("links") or {}).get("html") or ""
        meta = {"photographer": photographer, "profile_url": profile_url, "photo_url": photo_url, "unsplash_id": picked.get("id") or ""}
        items.append((img_bytes, photo_url or url_dl, meta))
    return items

def ensure_saved_candidate(img_bytes: Optional[bytes], out_path: Path, w: int, h: int,
                           focus_top: float, jitter: float) -> bool:
    if not img_bytes: return False
    try:
        save_resized(img_bytes, out_path, w, h, focus_top, jitter)
        return True
    except Exception:
        return False

def build_candidates(title: str, resume: str, video_url: str, base_dir: Path) -> Dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    bA, bB = choose_buckets(title, resume)

    vid = _parse_yt_id(video_url)
    igA_cands, igB_cands, fbA_cands, fbB_cands = [], [], [], []

    # IG — 1080x1350 (portrait)
    if vid and IMAGE_PROVIDER in {"hybrid"}:
        yt_url = yt_thumb_url(vid)
        b = download_with_retry(yt_url)
        if b and ensure_saved_candidate(b, base_dir/"IG_A_youtube.jpg", 1080, 1350, 0.30, 0.06):
            igA_cands.append({"source":"youtube","image_local":"images/IG_A_youtube.jpg","image_url":yt_url})

    kwA = ",".join(filter(None, [bA['unsplash'], bA.get('unsplash_extra','people,faces,editorial')]))
    for idx, (b, url, meta) in enumerate(unsplash_bytes_list(kwA, "1080x1350", title, "IG_A", 3, orientation="portrait"), start=1):
        name = f"IG_A_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1080, 1350, 0.30, 0.06):
            igA_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url,"attribution": meta})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bA["prompt_A"], size="1024x1536")
        if ensure_saved_candidate(b, base_dir/"IG_A_ai.jpg", 1080, 1350, 0.30, 0.06):
            igA_cands.append({"source":"openai","image_local":"images/IG_A_ai.jpg","image_url":""})

    kwB = ",".join(filter(None, [bB['unsplash'], bB.get('unsplash_extra','people,faces,editorial')]))
    for idx, (b, url, meta) in enumerate(unsplash_bytes_list(kwB, "1080x1350", title, "IG_B", 3, orientation="portrait"), start=1):
        name = f"IG_B_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1080, 1350, 0.28, 0.08):
            igB_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url,"attribution": meta})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bB["prompt_B"], size="1024x1536")
        if ensure_saved_candidate(b, base_dir/"IG_B_ai.jpg", 1080, 1350, 0.28, 0.08):
            igB_cands.append({"source":"openai","image_local":"images/IG_B_ai.jpg","image_url":""})

    # FB — 1200x630 (landscape)
    kwAfb = ",".join(filter(None, [bA['unsplash'], bA.get('unsplash_extra','people,faces,editorial')]))
    for idx, (b, url, meta) in enumerate(unsplash_bytes_list(kwAfb, "1600x1066", title, "FB_A", 3, orientation="landscape"), start=1):
        name = f"FB_A_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1200, 630, 0.40, 0.03):
            fbA_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url,"attribution": meta})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bA["prompt_A"], size="1536x1024")
        if ensure_saved_candidate(b, base_dir/"FB_A_ai.jpg", 1200, 630, 0.40, 0.03):
            fbA_cands.append({"source":"openai","image_local":"images/FB_A_ai.jpg","image_url":""})

    kwBfb = ",".join(filter(None, [bB['unsplash'], bB.get('unsplash_extra','people,faces,editorial')]))
    for idx, (b, url, meta) in enumerate(unsplash_bytes_list(kwBfb, "1600x1066", title, "FB_B", 3, orientation="landscape"), start=1):
        name = f"FB_B_u{idx}.jpg"
        if ensure_saved_candidate(b, base_dir/name, 1200, 630, 0.46, 0.05):
            fbB_cands.append({"source":"unsplash","image_local":f"images/{name}","image_url":url,"attribution": meta})
    if openai_client and IMAGE_PROVIDER in {"openai","hybrid"}:
        b = openai_image(bB["prompt_B"], size="1536x1024")
        if ensure_saved_candidate(b, base_dir/"FB_B_ai.jpg", 1200, 630, 0.46, 0.05):
            fbB_cands.append({"source":"openai","image_local":"images/FB_B_ai.jpg","image_url":""})

    def pick_selected(cands: List[Dict]) -> int:
        return 0 if cands else -1

    return {
        "ig": {"A": {"candidates": igA_cands, "selected": pick_selected(igA_cands)},
               "B": {"candidates": igB_cands, "selected": pick_selected(igB_cands)}},
        "fb": {"A": {"candidates": fbA_cands, "selected": pick_selected(fbA_cands)},
               "B": {"candidates": fbB_cands, "selected": pick_selected(fbB_cands)}},
    }

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

_INTERNAL_PAT = re.compile(r"(?i)\bwebinar|webinaire|visi[oô]|formation\s+interne|atelier|module|newsletter|youtube|vid[ée]o")

def _is_weak_caption(text: str, title_hint: str) -> bool:
    s = (text or "").strip()
    if len(s) == 0: return True
    if _INTERNAL_PAT.search(s): return True
    if (s.lower() == (title_hint or "").lower()) or (len(s) < 40 and "#" not in s): return True
    return False

def _write_if_missing_or_weak(path: Path, content: str, title_hint: str):
    content = (content or "").strip()
    if not path.exists():
        path.write_text(content, encoding="utf-8"); return
    try:
        cur = path.read_text(encoding="utf-8")
    except Exception:
        cur = ""
    if _is_weak_caption(cur, title_hint):
        path.write_text(content, encoding="utf-8")

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

        slug = _clean_filename(stem)
        droot = DRAFTS_DIR / slug
        (droot / "images").mkdir(parents=True, exist_ok=True)

        if SKIP_IF_COMPLETE and _is_complete_draft(droot):
            skipped += 1
            continue

        # captions
        caps = generate_captions(resume_text, title_hint, video_url)
        ig_caps = (caps.get("ig") or [])[:2]
        fb_caps = (caps.get("fb") or [])[:2]
        title   = caps.get("title") or _public_title(title_hint)

        def _safe_get(lst: List[str], i: int, fallback: str) -> str:
            return (lst[i] if i < len(lst) and (lst[i] or "").strip() else fallback)

        _write_if_missing_or_weak(droot / "ig_caption_A.txt", _safe_get(ig_caps, 0, title), title)
        _write_if_missing_or_weak(droot / "ig_caption_B.txt", _safe_get(ig_caps, 1, title), title)
        _write_if_missing_or_weak(droot / "fb_caption_A.txt", _safe_get(fb_caps, 0, title), title)
        _write_if_missing_or_weak(droot / "fb_caption_B.txt", _safe_get(fb_caps, 1, title), title)

        # images candidates (seulement si pas complet)
        cands = build_candidates(title, resume_text, video_url, droot / "images")

        # meta — merge idempotent et préservation des choix précédents
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
            for k1 in ("ig","fb"):
                if k1 in old:
                    for k2 in ("A","B"):
                        if k2 in old[k1] and old[k1][k2].get("candidates"):
                            meta[k1][k2]["candidates"] = old[k1][k2]["candidates"]
                            meta[k1][k2]["selected"]   = old[k1][k2].get("selected", 0)
            meta.update({k:v for k,v in old.items() if k not in meta})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] Brouillons (ré)générés dans: {DRAFTS_DIR.resolve()} — ignorés (déjà complets): {skipped}")

if __name__ == "__main__":
    main()
