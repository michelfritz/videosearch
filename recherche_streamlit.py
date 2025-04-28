
import streamlit as st
import pandas as pd
import os

# --- Chargement des bases ---
urls_df = pd.read_csv("urls.csv", encoding="utf-8")
urls_df = urls_df.sort_values(by="date", ascending=False)

# --- Nouveaux dossiers utilisés ---
DOSSIER_NEWSLETTERS = "newsletters"

# --- Fonction pour charger une newsletter HTML ---
def charger_newsletter_html(nom_fichier):
    chemin = os.path.join(DOSSIER_NEWSLETTERS, f"{nom_fichier}.html")
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None

# --- Fonction pour proposer un téléchargement de la newsletter ---
def bouton_telecharger_newsletter(nom_fichier, contenu_html):
    st.download_button(
        label="⬇️ Télécharger la Newsletter",
        data=contenu_html,
        file_name=f"{nom_fichier}.html",
        mime="text/html"
    )

# --- Affichage ---
st.title("Base de connaissance A LA LUCARNE")

for index, row in urls_df.iterrows():
    nom_fichier = row["fichier"]
    titre_video = row.get("titre", "Sans titre")
    date_video = row.get("date", "Date inconnue")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"{titre_video}")
        st.caption(f"📅 {date_video}")

    with col2:
        if st.button(f"📰 Voir Newsletter", key=f"newsletter_{index}"):
            newsletter_contenu = charger_newsletter_html(nom_fichier)
            if newsletter_contenu:
                with st.expander("📬 Newsletter liée à cette vidéo"):
                    st.markdown(newsletter_contenu, unsafe_allow_html=True)
                    bouton_telecharger_newsletter(nom_fichier, newsletter_contenu)
            else:
                st.warning("❌ Pas de newsletter disponible pour cette vidéo.")

    # Ici, tu affiches normalement ta recherche, résumés, tags, etc.
    # (on garde ton code existant en dessous)

    st.divider()
