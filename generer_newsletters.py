import os
import openai
from pathlib import Path

# --- Configuration ---
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
DOSSIER_SORTIE = "newsletters"
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Créer dossier sortie s'il n'existe pas ---
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# --- Prompt fusionné (éditorial + structure HTML) ---
PROMPT_BASE = (
    "Peux-tu me faire un résumé condensé des choses importantes qui ont été évoquées ? "
    "Passe les détails, effets comiques, répétitions et extraits inutiles. "
    "Essaye d'extraire uniquement des éléments utiles pour une newsletter, en particulier les faits marquants de la vie de notre réseau.\n\n"
    "Puis structure ta réponse en HTML :\n"
    "- Utilise un <h1> pour le titre principal.\n"
    "- Utilise un <h2> pour chaque grande section.\n"
    "- Utilise <p> pour les paragraphes.\n"
    "- Utilise des listes <ul><li> pour les faits marquants.\n"
    "- Ajoute quelques émojis 🎯🎉🚀 pour dynamiser.\n"
    "- Ne mets pas de balises <html> ni <body>, uniquement le contenu HTML."
)

# --- Ton bloc CSS sympa ---
BLOC_CSS = """
<style>
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    margin: 40px;
    color: #333333;
    background-color: #fafafa;
}
h1 {
    color: #1a73e8;
    text-align: center;
    font-size: 32px;
}
h2 {
    color: #0b5394;
    margin-top: 30px;
    font-size: 24px;
}
p {
    line-height: 1.6;
    margin-bottom: 20px;
}
ul {
    list-style-type: "🎯 ";
    margin-left: 20px;
    padding-left: 0;
}
li {
    margin-bottom: 10px;
    line-height: 1.5;
    font-size: 18px;
}
.container {
    background: white;
    padding: 40px;
    border-radius: 12px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
    max-width: 800px;
    margin: auto;
}
</style>
"""

# --- Fonction pour générer la newsletter via OpenAI ---
def generer_newsletter(texte_resume):
    prompt = f"{PROMPT_BASE}\n\nVoici la transcription :\n{texte_resume}"

    print("\n📝 Prompt envoyé (début) :")
    print(prompt[:1000])
    print("... (prompt tronqué pour affichage)\n")

    try:
        reponse = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        texte_brut = reponse.choices[0].message.content

        print("\n📨 Réponse reçue (début) :")
        print(texte_brut[:500])
        print("... (réponse tronquée pour affichage)\n")

        return texte_brut
    except Exception as e:
        print(f"❌ Erreur OpenAI détectée : {e}")
        return ""

# --- Traitement principal ---
print(f"🔎 Recherche de fichiers dans {DOSSIER_RESUME}...")
fichiers_trouves = list(Path(DOSSIER_RESUME).glob("*.txt"))
print(f"Fichiers trouvés : {[f.name for f in fichiers_trouves]}")

for fichier_txt in fichiers_trouves:
    nom_fichier = fichier_txt.stem
    print(f"\n✨ Traitement de : {nom_fichier}")

    with open(fichier_txt, "r", encoding="utf-8") as f:
        texte_resume = f.read()

    if texte_resume.strip():
        newsletter_brute = generer_newsletter(texte_resume)
        if newsletter_brute:
            contenu_final = f"{BLOC_CSS}\n<div class=\"container\">\n{newsletter_brute}\n</div>"
            fichier_html = Path(DOSSIER_SORTIE) / f"{nom_fichier}.html"
            with open(fichier_html, "w", encoding="utf-8") as f_out:
                f_out.write(contenu_final)
            print(f"📄 Newsletter stylée sauvegardée : {fichier_html}")
        else:
            print(f"⚠️ Aucun contenu généré pour {nom_fichier}")
    else:
        print(f"⚠️ Fichier vide : {nom_fichier}")

print("\n✅ Toutes les newsletters ont été générées et stylées.")
