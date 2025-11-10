# generate_presenter_script.py
# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime
from openai import OpenAI

# === Répertoire contenant les résumés ===
INPUT_DIR = r"C:\Transcript\Dropbox (Personal)\resume"
OUTPUT_DIR = INPUT_DIR  # on sauvegarde à côté
MODEL = "gpt-4o-mini"   # vous pouvez essayer "gpt-4o" si vous voulez un rendu plus léché

def read_all_summaries(input_dir: str) -> list[tuple[str, str]]:
    """Lit tous les .txt trouvés et retourne [(nom, contenu), ...] triés par nom."""
    paths = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    items = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    items.append((os.path.basename(p), txt))
        except UnicodeDecodeError:
            # fallback Windows-1252 si un fichier n'est pas en UTF-8
            with open(p, "r", encoding="cp1252", errors="replace") as f:
                txt = f.read().strip()
                if txt:
                    items.append((os.path.basename(p), txt))
    return items

def build_prompt(summaries: list[tuple[str, str]]) -> str:
    """Construit l’invite utilisateur avec tous les résumés."""
    blocks = []
    for name, content in summaries:
        blocks.append(f"### {name}\n{content}")
    joined = "\n\n---\n\n".join(blocks)

    # Consignes pour un texte oralisable fluide (~4–5 min)
    guidance = (
        "Tu es auteur·rice de voix-off. À partir des résumés ci-dessous, "
        "écris un script prêt à être lu à voix haute par un·e présentateur·rice, "
        "durée cible ~4 à 5 minutes (≈ 650 à 750 mots).\n\n"
        "Exigences :\n"
        "- Français naturel, fluide, sans jargon inutile.\n"
        "- Structure claire : accroche (1–2 phrases), contexte, 3–6 points clés avec transitions, "
        "micro-récap, conclusion avec appel à l’action (si pertinent).\n"
        "- Ton: accessible, sûr, dynamique mais posé (pas de surenchère).\n"
        "- Évite les listes exhaustives ; privilégie la narration et les enchaînements.\n"
        "- Pas d’indications de scène/caméra ; écris uniquement ce qui doit être DIT.\n"
        "- Si plusieurs vidéos se recoupent, fusionne sans redondances.\n"
        "- Pas d’invention de faits : reste fidèle aux résumés.\n"
        "- Orthographe/ponctuation soignées, phrases respirables pour l’oral.\n"
    )

    return guidance + "\n\n===== RÉSUMÉS =====\n" + joined

def main():
    summaries = read_all_summaries(INPUT_DIR)
    if not summaries:
        raise SystemExit(f"Aucun .txt trouvé dans : {INPUT_DIR}")

    user_prompt = build_prompt(summaries)

    client = OpenAI()  # utilise OPENAI_API_KEY depuis l’environnement

    # Appel simple via Responses API (SDK officiel)
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "Tu es un·e script doctor qui écrit des voix-off nettes et oralisables."},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Récupération du texte retourné (champ pratique du SDK)
    try:
        output_text = resp.output_text
    except Exception:
        # fallback si output_text n’est pas dispo
        if hasattr(resp, "output") and len(resp.output) > 0 and hasattr(resp.output[0], "content"):
            parts = []
            for p in resp.output[0].content:
                if hasattr(p, "text") and hasattr(p.text, "value"):
                    parts.append(p.text.value)
            output_text = "\n".join(parts).strip()
        else:
            raise RuntimeError("Réponse inattendue de l’API OpenAI.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUTPUT_DIR, f"script_presentateur_{ts}.md")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(output_text)

    print(f"✅ Script généré : {out_path}")

if __name__ == "__main__":
    main()
