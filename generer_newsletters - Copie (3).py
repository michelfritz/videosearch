# generer_newsletters.py — newsletter vivante + photo hero (OpenAI par défaut, Unsplash secours) + traçabilité source + email
# Version corrigée (08-11-2025) — fix stylize_sections: re.sub(pattern, repl, string)

import os
import re
import base64
import smtplib
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid

import requests  # fallback Unsplash

# ----------- Config dossiers -----------
DOSSIER_RESUME = os.getenv("DOSSIER_RESUME", r"C:\Transcript\Dropbox (Personal)\resume")
DOSSIER_SORTIE = os.getenv("DOSSIER_SORTIE", "newsletters")
DOSSIER_IMAGES = Path(DOSSIER_SORTIE) / "images"
os.makedirs(DOSSIER_SORTIE, exist_ok=True)
os.makedirs(DOSSIER_IMAGES, exist_ok=True)

# ----------- OpenAI (par défaut ON) -------------
USE_OPENAI_IMAGE = os.getenv("USE_OPENAI_IMAGE", "1") == "1"
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if (USE_OPENAI_IMAGE and OPENAI_API_KEY and OpenAI) else None

# ----------- Destinataires par défaut ---
DEFAULT_RECIPIENTS = [
    "michelfritz@alalucarne.com",
    "nicolasgasne@alalucarne.com",
]
RECIPIENTS = [
    e.strip() for e in os.getenv("NEWSLETTER_RECIPIENTS", ",".join(DEFAULT_RECIPIENTS)).split(",") if e.strip()
]

# ----------- SMTP config ----------------
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", "A La Lucarne <newsletter@alalucarne.com>")

# ----------- CSS -------------
BLOC_CSS = """
<style>
:root {
  --bg:#0b1220; --surface:#0f172a; --card:#111827; --txt:#e5e7eb; --muted:#9ca3af;
  --primary:#60a5fa; --accent:#34d399; --warn:#f59e0b;
}
*{box-sizing:border-box}
body {
  margin:0; padding:24px; background:linear-gradient(160deg, var(--bg), #0b1020 60%, #0b0f1a);
  color:var(--txt); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
}
.container {
  max-width: 900px; margin: 0 auto; background: rgba(17,24,39,.7);
  backdrop-filter: blur(6px); border:1px solid rgba(255,255,255,.06);
  border-radius: 16px; overflow:hidden; box-shadow: 0 20px 40px rgba(0,0,0,.35);
}
.hero { 
  position:relative; width:100%; height:320px; background:#0b1326; overflow:hidden;
}
@media (max-width:680px){ .hero{ height:220px; } }
.hero img { width:100%; height:100%; object-fit:cover; object-position:center 25%; display:block; }
.hero .overlay { position:absolute; inset:0; background:linear-gradient(180deg, rgba(0,0,0,.2), rgba(0,0,0,.65) 65%); }
.hero .title {
  position:absolute; left:24px; bottom:20px; right:24px;
  font-size: clamp(22px, 4vw, 36px); font-weight:800; line-height:1.15;
  color: #fff; text-shadow: 0 8px 20px rgba(0,0,0,.45);
}
.inner { padding: 28px; }
h1 { font-size: 26px; margin: 0 0 8px; color: var(--primary); }
h2 { font-size: 21px; margin: 26px 0 10px; color: #93c5fd;}
p { line-height:1.7; color: var(--txt); margin: 12px 0; }
ul { list-style: disc; padding-left: 1.25rem; margin: 12px 0; }
ul li { margin: 8px 0; }
.badges { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 20px;}
.badge { background:rgba(96,165,250,.18); border:1px solid rgba(96,165,250,.35); color:#dbeafe;
  padding:6px 10px; border-radius:999px; font-size:13px; }
.callout {
  border-left:4px solid var(--accent); background:rgba(52,211,153,.08); padding:14px 16px; border-radius:8px;
  margin:16px 0; color:#d1fae5;
}
.cta {
  display:inline-block; margin:16px 0 6px; background: linear-gradient(90deg,#38bdf8,#60a5fa);
  color:#0b1220; padding:10px 14px; font-weight:700; border-radius:10px; text-decoration:none;
  box-shadow: 0 8px 20px rgba(56,189,248,.35);
}
.footer { margin-top: 24px; color: var(--muted); font-size: 12px; text-align:center; }
code, pre { background: #0b1020; color: #e5e7eb; padding:3px 6px; border-radius:6px; }
@media (max-width:680px){ .inner{ padding:18px; } }
</style>
"""

# ----------- Prompt éditorial -------------
PROMPT_BASE = """Tu es un(e) rédacteur(trice) de newsletter interne pour un réseau immobilier (français).
Objectif: produire un contenu CLAIR, UTILE et ENJOUÉ, structuré en HTML (sans <html>/<body>), avec quelques émojis pertinents, sans excès.
Contraintes:
- Titre h1 accrocheur, concis.
- Sections h2 (proposées si adaptées): "À retenir", "Moments forts", "Ressources utiles", "À venir", "Bravo 👏", "Chiffres clés".
- Paragraphes courts (<p>), listes à puces (<ul><li>) avec verbes d’action.
- Ajoute 3 à 6 badges (mots-clés/hashtags) pertinents au-dessus du contenu principal (div.badges).
- Ajoute un bloc callout (div.callout) pour l’idée forte ou le conseil actionnable du jour.
- Termine par un CTA (lien) (ex: "Voir le replay", "Contacter le support formation").
- Pas d’emoji dans chaque ligne: privilégie pertinence.
- RENDS UNIQUEMENT LE HTML DU CONTENU (sans <html>/<body>) ET SANS BLOCS DE CODE (pas de ```html ni ```).

Réécris uniquement le contenu éditorial en HTML (sans style ni <body>), sur la base de la transcription suivante (résumé brut) :"""

# ----------- Helpers titres/sections/sanitization -----------
def extract_h1(html: str) -> str:
    if not html:
        return "Newsletter A La Lucarne"
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return "Newsletter A La Lucarne"
    title = re.sub(r"<.*?>", "", m.group(1)).strip()
    return title or "Newsletter A La Lucarne"

def stylize_sections(html: str) -> str:
    """Fixe: utiliser re.sub(pattern, repl, string). Ajout d'un garde si html est None."""
    if not html:
        return ""
    patterns = [
        (r"(?i)<h2[^>]*>\s*(?:à|a)\s*retenir\s*</h2>",                  "<h2>🧠 À retenir</h2>"),
        (r"(?i)<h2[^>]*>\s*moments?\s+forts?\s*</h2>",                  "<h2>⚡ Moments forts</h2>"),
        (r"(?i)<h2[^>]*>\s*ressources?\s+utiles?\s*</h2>",              "<h2>📚 Ressources utiles</h2>"),
        (r"(?i)<h2[^>]*>\s*(?:à|a)\s*venir\s*</h2>",                    "<h2>📅 À venir</h2>"),
        (r"(?i)<h2[^>]*>\s*chiffres?\s+cl(?:é|e|ee|ef|efs|es)\s*</h2>", "<h2>📊 Chiffres clés</h2>"),
    ]
    for pat, rep in patterns:
        html = re.sub(pat, rep, html)
    return html

def strip_code_fences(html: str) -> str:
    h = (html or "").strip()
    h = re.sub(r"^```(?:html|HTML)?\s*", "", h)
    h = re.sub(r"\s*```$", "", h)
    h = re.sub(r"```(?:html|HTML)?", "", h)
    h = re.sub(r"```", "", h)
    return h.strip()

# ----------- Génération newsletter ----------
def generer_newsletter(texte_resume: str) -> str:
    if not client:
        return f"<h1>Newsletter</h1><div class='badges'></div><div class='callout'>Résumé non réécrit.</div><p>{(texte_resume or '')[:800]}...</p><a class='cta' href='#'>Voir le replay</a>"
    prompt = f"{PROMPT_BASE}\n\n{texte_resume}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )
        html = resp.choices[0].message.content
        return strip_code_fences(html)
    except Exception as e:
        print("[WARN] OpenAI chat error:", e)
        return f"<h1>Newsletter</h1><div class='badges'></div><div class='callout'>Résumé non réécrit.</div><p>{(texte_resume or '')[:800]}...</p><a class='cta' href='#'>Voir le replay</a>"

# ----------- Image génération / fallback -------------
FRENCH_STOPWORDS = {
    "le","la","les","un","une","des","de","du","d","et","ou","au","aux","en","dans","sur","pour",
    "avec","sans","par","vers","chez","se","sa","son","ses","ce","cet","cette","ces","que","qui",
    "quoi","dont","où","quand","comme","plus","moins","très","tres","bien","a","à","the","of"
}

def keywords_from_prompt(prompt: str, max_terms: int = 6) -> str:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", (prompt or "").lower())
    clean, seen = [], set()
    seeds = ["photography","realistic","editorial","business","real estate","people","faces","meeting"]
    for w in seeds:
        if w not in seen:
            clean.append(w); seen.add(w)
    for w in words:
        if len(w) <= 3 or w in FRENCH_STOPWORDS:
            continue
        if w not in seen:
            seen.add(w); clean.append(w)
        if len(clean) >= max_terms:
            break
    return ",".join(clean)

def download_unsplash_image(prompt: str, out_path: Path) -> Tuple[Optional[Path], str]:
    """Unsplash Source (pas d'API key). Retourne (path, source_label)."""
    try:
        query = keywords_from_prompt(prompt)
        url = f"https://source.unsplash.com/1536x1024/?{urllib.parse.quote(query)}"
        r = requests.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200 and r.content:
            out_path.write_bytes(r.content)
            src = f"Unsplash ({query})"
            print(f"[OK] Unsplash image for '{query}' -> {out_path.name}")
            return out_path, src
        print(f"[WARN] Unsplash returned status {r.status_code} for query='{query}'")
    except Exception as e:
        print("[WARN] Unsplash download failed:", e)
    return None, ""

def generer_openai_image(prompt: str, out_path: Path) -> Tuple[Optional[Path], str]:
    if not (client and prompt):
        return None, ""
    enriched = (
        prompt
        + " — editorial, photorealistic office scene with real people, faces clearly visible, mid-shot, natural lighting,"
          " shallow depth of field, professional color grading;"
          " no illustration, no painting, no cartoon, no 3d, no cgi, no render, no watermark, no text"
    )
    sizes = ["1536x1024", "1024x1024", "1024x1536", "auto"]
    for sz in sizes:
        try:
            im = client.images.generate(
                model="gpt-image-1",
                prompt=enriched,
                size=sz,
                quality="high",
                n=1,
            )
            b64 = im.data[0].b64_json
            data = base64.b64decode(b64)
            out_path.write_bytes(data)
            src = f"OpenAI gpt-image-1 ({sz})"
            print(f"[OK] OpenAI image size={sz} -> {out_path.name}")
            return out_path, src
        except Exception as e:
            print(f"[WARN] OpenAI image failed size={sz}: {e}")
    return None, ""

def generer_image(prompt: str, out_path: Path) -> Tuple[Optional[Path], str]:
    """OpenAI prioritaire (faces visibles), sinon Unsplash. Retourne (path, source_label)."""
    img, src = generer_openai_image(prompt, out_path)
    if img:
        return img, src
    return download_unsplash_image(prompt, out_path)

# ----------- Assemblage HTML final ----------
def assemble_html(full_html_inner: str, title_fallback: str, hero_file: Optional[Path], image_source: str = "") -> str:
    title = extract_h1(full_html_inner) or title_fallback
    inner_no_h1 = re.sub(r"<h1[^>]*>.*?</h1>", "", full_html_inner or "", flags=re.IGNORECASE | re.DOTALL).strip()

    source_comment = f"<!-- image-source: {image_source} -->\n" if image_source else ""

    hero_tag = ""
    if hero_file and hero_file.exists():
        hero_src = f"images/{hero_file.name}"
        hero_tag = (
            f'<div class="hero">'
            f'<img src="{hero_src}" alt="Hero" data-source="{image_source}"><div class="overlay"></div>'
            f'<div class="title">{title}</div>'
            f'</div>'
        )
    h1_block = "" if hero_tag else f"<h1>{title}</h1>"

    html = f"""{BLOC_CSS}
{source_comment}<div class="container">
  {hero_tag}
  <div class="inner">
    {h1_block}
    {inner_no_h1}
    <div class="footer">
      Newsletter générée automatiquement – A La Lucarne • {formatdate(localtime=True)}
    </div>
  </div>
</div>"""
    return html

# ----------- Email (HTML + CID image) -------
def html_to_text(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html or "", flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<li>", "- ", text, flags=re.I)
    text = re.sub(r"<.*?>", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def send_email(subject: str, html: str, recipients: list[str], hero_bytes: Optional[bytes]):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM):
        print("[INFO] SMTP non configuré -> envoi email ignoré.")
        return

    msg_root = MIMEMultipart("related")
    msg_root["Subject"] = subject
    msg_root["From"] = SMTP_FROM
    msg_root["To"] = ", ".join(recipients)
    msg_root["Date"] = formatdate(localtime=True)

    alt = MIMEMultipart("alternative")
    msg_root.attach(alt)

    text = html_to_text(html or "")
    alt.attach(MIMEText(text, "plain", "utf-8"))

    if hero_bytes:
        cid = make_msgid(domain="alalucarne.com")[1:-1]
        img = MIMEImage(hero_bytes)
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename="hero.png")
        msg_root.attach(img)
        html_email = re.sub(r'src="images/[^"]+"', f'src="cid:{cid}"', html or "")
    else:
        html_email = html or ""

    alt.attach(MIMEText(html_email, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_FROM, recipients, msg_root.as_string())
    print("[OK] Email envoyé à:", ", ".join(recipients))

# ----------- MAIN ---------------------------
def main():
    print("[INFO] Recherche de résumés TXT dans:", DOSSIER_RESUME)
    fichiers = list(Path(DOSSIER_RESUME).glob("*.txt"))
    print("[INFO] Trouvés:", [f.name for f in fichiers])

    for fichier_txt in fichiers:
        nom = fichier_txt.stem
        print("\n[RUN] Construction newsletter pour:", nom)

        texte_resume = fichier_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not texte_resume:
            print("[WARN] Fichier vide, on passe:", nom)
            continue

        inner_html = generer_newsletter(texte_resume)
        inner_html = stylize_sections(inner_html)
        title_fallback = "Newsletter A La Lucarne"

        # Image Hero
        img_prompt = f"{nom}: real estate business meeting, collaborative team, faces visible, professional office, candid smiles"
        out_path = (DOSSIER_IMAGES / f"{nom}.png")
        hero_file, image_source = generer_image(img_prompt, out_path)

        hero_bytes: Optional[bytes] = None
        if hero_file and hero_file.exists():
            hero_bytes = hero_file.read_bytes()
            print(f"[OK] Image hero prête via: {image_source}")
        else:
            print("[INFO] Pas d'image hero générée.")
            image_source = ""

        final_html = assemble_html(inner_html, title_fallback, hero_file, image_source)

        out_html = Path(DOSSIER_SORTIE) / f"{nom}.html"
        out_html.write_text(final_html, encoding="utf-8")
        print("[OK] Newsletter sauvegardée:", out_html)

        subject = extract_h1(inner_html) or f"Newsletter - {nom}"
        try:
            send_email(subject, final_html, RECIPIENTS, hero_bytes)
        except Exception as e:
            print("[WARN] Erreur d'envoi email:", e)

    print("\n[OK] Toutes les newsletters sont générées.")

if __name__ == "__main__":
    main()
