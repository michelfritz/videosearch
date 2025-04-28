import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")

openai.api_key = st.secrets["OPENAI_API_KEY"]

DOSSIER_NEWSLETTERS = "newsletters"

# --- Fonctions newsletters ---
def charger_newsletter_html(nom_fichier):
    chemin = os.path.join(DOSSIER_NEWSLETTERS, f"{nom_fichier}.html")
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None

def bouton_telecharger_newsletter(nom_fichier, contenu_html):
    st.download_button(
        label="⬇️ Télécharger la Newsletter",
        data=contenu_html,
        file_name=f"{nom_fichier}.html",
        mime="text/html"
    )

@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls_et_idees_themes_sujets():
    try:
        urls = pd.read_csv("urls.csv", encoding="utf-8")
    except UnicodeDecodeError:
        urls = pd.read_csv("urls.csv", encoding="cp1252")
    urls["titre"] = urls["titre"].fillna("Titre inconnu")
    urls["date"] = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")
    urls["fichier"] = urls["fichier"].str.replace(".json", "", regex=False)

    try:
        idees_v2 = pd.read_csv("idees_v2.csv", encoding="utf-8")
    except UnicodeDecodeError:
        idees_v2 = pd.read_csv("idees_v2.csv", encoding="cp1252")
    idees_v2_grouped = idees_v2.groupby("fichier").apply(lambda x: x.to_dict(orient="records")).reset_index()
    idees_v2_grouped.columns = ["fichier", "idees_v2"]

    try:
        idees = pd.read_csv("idees.csv", encoding="utf-8")
    except UnicodeDecodeError:
        idees = pd.read_csv("idees.csv", encoding="cp1252")

    if "idee" in idees.columns:
        idees_grouped = idees.groupby("fichier").apply(lambda x: x["idee"].dropna().tolist()).reset_index()
        idees_grouped.columns = ["fichier", "sujets"]
    else:
        idees_grouped = pd.DataFrame(columns=["fichier", "sujets"])

    try:
        themes = pd.read_csv("themes.csv", encoding="utf-8")
    except UnicodeDecodeError:
        themes = pd.read_csv("themes.csv", encoding="cp1252")
    themes["themes"] = themes["themes"].fillna("")

    df = pd.merge(urls, idees_v2_grouped, on="fichier", how="left")
    df = pd.merge(df, idees_grouped, on="fichier", how="left")
    df = pd.merge(df, themes, on="fichier", how="left")
    return df

# --- Interface principale ---
st.title("📚 Base de connaissance A LA LUCARNE")

df, vecteurs = charger_donnees()
urls_df = charger_urls_et_idees_themes_sujets()

menu = st.sidebar.radio("Navigation", ["🔍 Recherche", "🎥 Toutes les vidéos"])

if menu == "🔍 Recherche":
    query = st.text_input("🧐 Que veux-tu savoir ?", "")
    seuil = st.slider("🎯 Exigence des résultats", 0.1, 0.9, 0.5, 0.05)

    if query:
        vecteur_query = embed_openai(query)
        indices, scores = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)

        if len(indices) == 0:
            st.warning("Aucun résultat trouvé.")
        else:
            for idx, score in zip(indices, scores):
                bloc = df.iloc[idx]
                url_complet = bloc["url"]
                youtube_id = ""
                if "watch?v=" in url_complet:
                    youtube_id = url_complet.split("watch?v=")[-1]
                elif "youtu.be/" in url_complet:
                    youtube_id = url_complet.split("youtu.be/")[-1]

                start_time = int(float(bloc.get("start", 0)))
                embed_url = f"https://www.youtube.com/embed/{youtube_id}?start={start_time}&autoplay=0"

                with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}..."):
                    st.markdown(f"**Texte complet :** {bloc['text']}")
                    if youtube_id:
                        st.components.v1.iframe(embed_url, height=315)

elif menu == "🎥 Toutes les vidéos":
    recherche = st.text_input("🔍 Rechercher une vidéo", "")
    tri = st.selectbox("Trier par", ("Date récente", "Date ancienne", "Titre A → Z", "Titre Z → A"))

    if recherche:
        urls_df = urls_df[urls_df.apply(lambda row: recherche.lower() in (str(row["titre"])+str(row["resume"])+str(row.get("themes", ""))).lower(), axis=1)]

    if tri == "Date récente":
        urls_df = urls_df.sort_values("date", ascending=False)
    elif tri == "Date ancienne":
        urls_df = urls_df.sort_values("date", ascending=True)
    elif tri == "Titre A → Z":
        urls_df = urls_df.sort_values("titre", ascending=True)
    elif tri == "Titre Z → A":
        urls_df = urls_df.sort_values("titre", ascending=False)

    for _, row in urls_df.iterrows():
        video_name = row.get("titre", "Titre inconnu")
        video_date = row.get("date", "Date inconnue")
        url_complet = row.get("url", "")
        resume = row.get("resume", "")
        themes = row.get("themes", "")
        idees_v2 = row.get("idees_v2", [])
        sujets = row.get("sujets", [])
        fichier_nom = row.get("fichier", "")

        if "watch?v=" in url_complet:
            youtube_id = url_complet.split("watch?v=")[-1]
        elif "youtu.be/" in url_complet:
            youtube_id = url_complet.split("youtu.be/")[-1]
        else:
            youtube_id = ""

        thumbnail_url = f"https://img.youtube.com/vi/{youtube_id}/0.jpg"

        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(thumbnail_url, width=140)
        with col2:
            st.markdown(f"### [{video_name}]({url_complet})")
            st.markdown(f"🗓️ *{video_date}*")
            if resume:
                st.markdown(f"📜 {resume}")

            if fichier_nom:
                if st.button("📰 Voir Newsletter", key=f"newsletter_{fichier_nom}"):
                    newsletter_contenu = charger_newsletter_html(fichier_nom)
                    if newsletter_contenu:
                        with st.expander("📬 Newsletter liée à cette vidéo"):
                            st.markdown(newsletter_contenu, unsafe_allow_html=True)
                            bouton_telecharger_newsletter(fichier_nom, newsletter_contenu)
                    else:
                        st.warning("❌ Pas de newsletter disponible.")

            if themes:
                tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
                for theme in themes.split("|"):
                    theme = theme.strip()
                    if theme:
                        tags_html += f"<span style='background-color: #e1e4e8; padding: 5px 10px; border-radius: 15px;'>{theme}</span>"
                tags_html += "</div>"
                st.markdown(tags_html, unsafe_allow_html=True)

            if not isinstance(sujets, list):
                sujets = []

            if sujets:
                with st.expander("📚 Sujets principaux de la vidéo", expanded=False):
                    for sujet in sujets:
                        st.markdown(f"- {sujet}")

            if idees_v2:
                with st.expander("🌟 Grands moments de la vidéo"):
                    for idee_obj in idees_v2:
                        idee = idee_obj.get("idee", "")
                        start = idee_obj.get("start", 0)
                        if idee and youtube_id:
                            st.markdown(f"- [{idee}](https://www.youtube.com/watch?v={youtube_id}&t={start}s)")
                        elif idee:
                            st.markdown(f"- {idee}")

        st.markdown("---")
