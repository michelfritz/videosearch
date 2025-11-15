# -*- coding: utf-8 -*-
# generer_newsletters.py — Incrémental robuste via sentinelles + registre

import os, re, base64, smtplib, urllib.parse
from pathlib import Path
from typing import Optional, Tuple
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid

import requests
from incremental_utils import compute_fingerprint, should_skip, mark_done, backfill_if_up_to_date

# ----------- Config dossiers -----------
DOSSIER_RESUME = os.getenv("DOSSIER_RESUME", r"C:\Transcript\Dropbox (Personal)\resume")
DOSSIER_SORTIE = os.getenv("DOSSIER_SORTIE", "newsletters")
DOSSIER_IMAGES = Path(DOSSIER_SORTIE) / "images"
Path(DOSSIER_SORTIE).mkdir(parents=True, exist_ok=True)
Path(DOSSIER_IMAGES).mkdir(parents=True, exist_ok=True)

SCRIPT_NAME = "newsletters"

# ----------- OpenAI (optionnel) -------------
USE_OPENAI_IMAGE = os.getenv("USE_OPENAI_IMAGE", "1") == "1"
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=OPENAI_API_KEY) if (USE_OPENAI_IMAGE and OPENAI_API_KEY) else None
except Exception:
    client = None

# ----------- Destinataires ---
DEFAULT_RECIPIENTS = [
    "nicolasgasne@alalucarne.com",
    "michelcharlesfritz@gmail.com",
    "newsletters.b10ait@zapiermail.com",
]

RECIPIENTS = [
    e.strip()
    for e in os.getenv("NEWSLETTER_RECIPIENTS", ",".join(DEFAULT_RECIPIENTS)).split(",")
    if e.strip()
]

# ----------- SMTP (Mailjet) ----------------
MAILJET_API_KEY = os.getenv("MAILJET_API_KEY")
MAILJET_SECRET_KEY = os.getenv("MAILJET_SECRET_KEY")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM")

# Si aucun SMTP explicite n'est défini mais que Mailjet est configuré,
# on utilise automatiquement le relais SMTP Mailjet.
if not SMTP_HOST and MAILJET_API_KEY and MAILJET_SECRET_KEY:
    SMTP_HOST = "in-v3.mailjet.com"  # relais SMTP Mailjet
    SMTP_USER = MAILJET_API_KEY
    SMTP_PASS = MAILJET_SECRET_KEY

# Expéditeur par défaut
if not SMTP_FROM:
    SMTP_FROM = "A La Lucarne <newsletter@alalucarne.com>"

# ----------- CSS -------------
BLOC_CSS = """
<style>
:root { --bg:#0b1220; --surface:#0f172a; --card:#111827; --txt:#e5e7eb; --muted:#9ca3af; --primary:#60a5fa; --accent:#34d399; }
*{box-sizing:border-box}
body{ margin:0; padding:24px; background:linear-gradient(160deg, var(--bg), #0b1020 60%, #0b0f1a); color:var(--txt); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;}
.container{ max-width: 900px; margin:0 auto; background:rgba(17,24,39,.7); border:1px solid rgba(255,255,255,.06); border-radius:16px; overflow:hidden; }
.hero{ position:relative; width:100%; height:320px; background:#0b1326; overflow:hidden; } @media (max-width:680px){ .hero{ height:220px; } }
.hero img{ width:100%; height:100%; object-fit:cover; object-position:center 25%; display:block; }
.hero .overlay{ position:absolute; inset:0; background:linear-gradient(180deg, rgba(0,0,0,.2), rgba(0,0,0,.65) 65%); }
.hero .title{ position:absolute; left:24px; bottom:20px; right:24px; font-size: clamp(22px, 4vw, 36px); font-weight:800; line-height:1.15; color:#fff; }
.inner{ padding:28px; } h1{ font-size:26px; margin:0 0 8px; color:var(--primary);} h2{ font-size:21px; margin:26px 0 10px; color:#93c5fd;} p{ line-height:1.7; margin:12px 0;}
ul{ list-style:disc; padding-left:1.25rem; margin:12px 0;} ul li{ margin:8px 0;}
.badges{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 20px;} .badge{ background:rgba(96,165,250,.18); border:1px solid rgba(96,165,250,.35); color:#dbeafe; padding:6px 10px; border-radius:999px; font-size:13px; }
.callout{ border-left:4px solid var(--accent); background:rgba(52,211,153,.08); padding:14px 16px; border-radius:8px; margin:16px 0; color:#d1fae5; }
.cta{ display:inline-block; margin:16px 0 6px; background:linear-gradient(90deg,#38bdf8,#60a5fa); color:#0b1220; padding:10px 14px; font-weight:700; border-radius:10px; text-decoration:none; }
.footer{ margin-top:24px; color:#9ca3af; font-size:12px; text-align:center; }
</style>
"""

PROMPT_BASE = """Tu es un(e) rédacteur(trice) de newsletter interne pour un réseau immobilier (français).

IMPORTANT — ANCRAGE SUR LA SOURCE :
- Tu dois te baser STRICTEMENT sur le texte ci-dessous (résumé issu d’une vidéo interne).
- Tu N’AJOUTES AUCUNE information qui n’est pas explicitement présente dans ce texte.
- Tu n’inventes pas de chiffres, dates, exemples, citations, lois, offres commerciales ou scénarios.
- Si un élément est absent ou flou dans le texte, tu restes vague ou tu n’en parles pas.
- Tu ne fais PAS de généralités sur le marché immobilier ou l’actualité nationale si ce n’est pas mentionné.

STYLE GÉNÉRAL :
- Tu écris une newsletter envoyée APRÈS la visio, pas un script parlé de visio en direct.
- Tu ne commences PAS par une formule de salutation (“Bonjour à tous”, “Salut”, etc.).
- Tu évites “Aujourd’hui nous allons…”, “Dans cette visio…”. Tu préfères des phrases de synthèse.
- Tu gardes un ton professionnel, clair, dynamique, mais pas trop familier.

INTRODUCTION :
- Le premier paragraphe doit être une courte phrase de contexte qui résume l’enjeu principal de la visio
  à partir du texte source. Par exemple :
  - “Les questions remontées en préparation de la dernière visio étaient nombreuses,
     mais le point essentiel était celui des logements décents.”
- Tu NE recopies PAS les phrases d’introduction de type “Bonjour à tous…” présentes dans le texte source.
- Tu transformes ces passages d’ouverture en une synthèse écrite adaptée à une newsletter.

Objectif : produire un contenu CLAIR, UTILE et ENJOUÉ, structuré en HTML (sans <html>/<body>), avec quelques émojis pertinents, sans excès.

Contraintes de forme :
- Titre <h1> accrocheur, concis, fidèle au contenu du texte.
- Sections <h2> (uniquement si elles se justifient par le texte) possibles : "À retenir", "Moments forts", "Ressources utiles", "À venir", "Bravo 👏", "Chiffres clés".
- Paragraphes courts (<p>), listes à puces (<ul><li>) avec verbes d’action, SANS ajout d’informations externes.
- Ajoute 3 à 6 badges (mots-clés/hashtags) dans <div class="badges"> à partir des thèmes réellement présents dans le texte.
- Ajoute un bloc <div class="callout"> pour l’idée forte ou le conseil actionnable, directement issu du texte.
- Termine par un CTA (lien) cohérent avec le texte (ex: "Voir le replay", "Télécharger le support" si mentionné ; sinon un CTA neutre comme "Contacter le service formation").

Consignes supplémentaires :
- Tu ne mentionnes pas d’outils, de process, de chiffres réseau ou d’éléments internes qui ne figurent pas dans le texte.
- Pas de blocs de code, pas de ```html ni ``` dans la réponse.
- RENDS UNIQUEMENT LE HTML DU CONTENU (sans <html>/<body>)."""

PROMPT_TAIL = """
Réécris uniquement le contenu éditorial en HTML (sans style ni <body>), sur la base de la transcription suivante :
"""

def extract_h1(html: str) -> str:
    if not html:
        return "Newsletter A La Lucarne"
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.I | re.S)
    if not m:
        return "Newsletter A La Lucarne"
    t = re.sub(r"<.*?>", "", m.group(1)).strip()
    return t or "Newsletter A La Lucarne"

def stylize_sections(html: str) -> str:
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

def generer_newsletter(texte_resume: str) -> str:
    if not client:
        return (
            "<h1>Newsletter</h1><div class='badges'></div>"
            "<div class='callout'>Résumé non réécrit.</div>"
            f"<p>{(texte_resume or '')[:800]}...</p>"
            "<a class='cta' href='#'>Voir le replay</a>"
        )
    prompt = f"{PROMPT_BASE}\n\n{PROMPT_TAIL}\n{texte_resume}"
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
        return (
            "<h1>Newsletter</h1><div class='badges'></div>"
            "<div class='callout'>Résumé non réécrit.</div>"
            f"<p>{(texte_resume or '')[:800]}...</p>"
            "<a class='cta' href='#'>Voir le replay</a>"
        )

def keywords_from_prompt(prompt: str, max_terms: int = 6) -> str:
    STOP = {
        "le","la","les","un","une","des","de","du","d","et","ou","au","aux","en","dans",
        "sur","pour","avec","sans","par","vers","chez","se","sa","son","ses","ce","cet",
        "cette","ces","que","qui","quoi","dont","où","quand","comme","plus","moins",
        "très","tres","bien","a","à","the","of"
    }
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", (prompt or "").lower())
    clean, seen = [], set()
    seeds = ["photography", "realistic", "editorial", "business", "real estate", "people", "faces", "meeting"]
    for w in seeds:
        if w not in seen:
            clean.append(w); seen.add(w)
    for w in words:
        if len(w) <= 3 or w in STOP:
            continue
        if w not in seen:
            seen.add(w); clean.append(w)
        if len(clean) >= max_terms:
            break
    return ",".join(clean)

def download_unsplash_image(prompt: str, out_path: Path):
    try:
        query = keywords_from_prompt(prompt)
        url = f"https://source.unsplash.com/1536x1024/?{urllib.parse.quote(query)}"
        r = requests.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200 and r.content:
            out_path.write_bytes(r.content)
            return out_path, f"Unsplash ({query})"
    except Exception as e:
        print("[WARN] Unsplash download failed:", e)
    return None, ""

def generer_openai_image(prompt: str, out_path: Path):
    if not client:
        return (None, "")
    enriched = (
        prompt
        + " — editorial, photorealistic office scene with real people, faces clearly visible, "
          "natural lighting, professional color grading; no illustration, no cartoon, no 3d, "
          "no cgi, no render, no watermark, no text"
    )
    sizes = ["1536x1024", "1024x1024", "1024x1536"]
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
            return out_path, f"OpenAI gpt-image-1 ({sz})"
        except Exception as e:
            print(f"[WARN] OpenAI image failed size={sz}: {e}")
    return None, ""

def generer_image(prompt: str, out_path: Path):
    img, src = generer_openai_image(prompt, out_path)
    if img:
        return img, src
    return download_unsplash_image(prompt, out_path)

def assemble_html(full_html_inner: str, title_fallback: str, hero_file: Optional[Path], image_source: str = "") -> str:
    title = extract_h1(full_html_inner) or title_fallback
    inner_no_h1 = re.sub(r"<h1[^>]*>.*?</h1>", "", full_html_inner or "", flags=re.I | re.S).strip()

    hero_tag = ""
    if hero_file and hero_file.exists():
        hero_src = f"images/{hero_file.name}"
        hero_tag = (
            f'<div class="hero"><img src="{hero_src}" alt="Hero" data-source="{image_source}">'
            f'<div class="overlay"></div><div class="title">{title}</div></div>'
        )

    h1_block = "" if hero_tag else f"<h1>{title}</h1>"

    return f"""{BLOC_CSS}
<div class="container">
  {hero_tag}
  <div class="inner">
    {h1_block}
    {inner_no_h1}
    <div class="footer">Newsletter &ndash; A La Lucarne &bull; {formatdate(localtime=True)}</div>
  </div>
</div>"""

def html_to_text(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html or "", flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<li>", "- ", text, flags=re.I)
    text = re.sub(r"<.*?>", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def send_email(subject: str, html: str, recipients: list[str], hero_bytes: Optional[bytes]):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM):
        print("[INFO] SMTP non configuré -> email non envoyé.")
        return

    msg_root = MIMEMultipart("related")
    msg_root["Subject"] = subject
    msg_root["From"] = SMTP_FROM
    msg_root["To"] = ", ".join(recipients)
    msg_root["Date"] = formatdate(localtime=True)

    alt = MIMEMultipart("alternative")
    msg_root.attach(alt)

    text = html_to_text(html)
    alt.attach(MIMEText(text, "plain", "utf-8"))

    if hero_bytes:
        cid = make_msgid(domain="alalucarne.com")[1:-1]
        img = MIMEImage(hero_bytes)
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename="hero.png")
        msg_root.attach(img)
        html = re.sub(r'src="images/[^"]+"', f'src="cid:{cid}"', html)

    alt.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_FROM, recipients, msg_root.as_string())

    print("[OK] Email envoyé à:", ", ".join(recipients))

def main():
    print("[INFO] Recherche de résumés TXT dans:", DOSSIER_RESUME)
    fichiers = list(Path(DOSSIER_RESUME).glob("*.txt"))
    print("[INFO] Trouvés:", [f.name for f in fichiers])

    for fichier_txt in fichiers:
        nom = fichier_txt.stem
        txt = fichier_txt.read_text(encoding="utf-8", errors="ignore")
        fp = compute_fingerprint(nom, txt)  # clé = nom ; empreinte = contenu

        out_html = Path(DOSSIER_SORTIE) / f"{nom}.html"

        # Backfill de sentinelle si le HTML est déjà plus récent que le TXT (premier run)
        if backfill_if_up_to_date(SCRIPT_NAME, nom, fp, fichier_txt, out_html):
            if os.getenv("SKIP_IF_UP_TO_DATE", "1") == "1":
                print(f"[SKIP] À jour (mtime) -> {nom}")
                continue

        # Skip si sentinelle correspondante
        if should_skip(SCRIPT_NAME, nom, fp):
            print(f"[SKIP] Déjà traité (fingerprint ok) -> {nom}")
            continue

        print("\n[RUN] Construction newsletter pour:", nom)
        inner_html = generer_newsletter(txt.strip())
        inner_html = stylize_sections(inner_html)
        title_fallback = "Newsletter A La Lucarne"

        img_prompt = f"{nom}: real estate business meeting, collaborative team, faces visible, professional office, candid smiles"
        out_path = Path(DOSSIER_IMAGES) / f"{nom}.png"
        hero_file, image_source = generer_image(img_prompt, out_path)

        hero_bytes: Optional[bytes] = None
        if hero_file and hero_file.exists():
            hero_bytes = hero_file.read_bytes()
            print(f"[OK] Image hero prête via: {image_source}")
        else:
            print("[INFO] Pas d'image hero générée.")
            image_source = ""

        final_html = assemble_html(inner_html, title_fallback, hero_file, image_source)
        out_html.write_text(final_html, encoding="utf-8")
        print("[OK] Newsletter sauvegardée:", out_html)

        subject = extract_h1(inner_html) or f"Newsletter - {nom}"
        try:
            send_email(subject, final_html, RECIPIENTS, hero_bytes)
        except Exception as e:
            print("[WARN] Erreur d'envoi email:", e)

        # Marquer comme fait
        mark_done(
            SCRIPT_NAME,
            nom,
            str(fichier_txt),
            fp,
            str(out_html),
            extra={"image_source": image_source},
        )

    print("\n[OK] Toutes les newsletters sont à jour.")

if __name__ == "__main__":
    main()
