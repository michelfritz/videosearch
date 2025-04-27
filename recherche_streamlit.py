import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai

st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")

# 🔐 Clé API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📚 Charger les données
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls_idees_themes():
    urls = pd.read_csv("urls.csv", encoding="utf-8")
    themes = pd.read_csv("themes.csv", encoding="utf-8")
    idees = pd.read_csv("idees_v2.csv", encoding="utf-8")

    urls["titre"] = urls["titre"].fillna("Titre inconnu")
    urls["date"] = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")

    # Regrouper les idées par fichier
    idees_grouped = idees.groupby("fichier").apply(lambda x: x.to_dict(orient="records")).reset_index()
    idees_grouped.columns = ["fichier", "idees"]

    df = pd.merge(urls, themes, on="fichier", how="left")
    df = pd.merge(df, idees_grouped, on="fichier", how="left")
    return df

# 🔎 Embedding OpenAI
def embed_openai(query):
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small",
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

# 🔥 Recherche de similarité
def rechercher_similaires(vecteur_query, vecteurs, top_k=5, seuil=0.3):
    similarities = np.dot(vecteurs, vecteur_query)
    indices = np.where(similarities >= seuil)[0]
    top_indices = indices[np.argsort(similarities[indices])[::-1][:top_k]]
    return top_indices, similarities[top_indices]

# 🛠 Interface Streamlit
st.title("📚 Base de connaissance A LA LUCARNE")

df, vecteurs = charger_donnees()
urls_df = charger_urls_idees_themes()

# 📂 Menu latéral
menu = st.sidebar.radio("Navigation", ["🔍 Recherche", "🎥 Toutes les vidéos"])

if menu == "🎥 Toutes les vidéos":
    st.header("📚 Liste des vidéos disponibles")

    recherche = st.text_input("🔎 Recherche par titre, résumé, idée ou thème", "")

    tri = st.selectbox(
        "📜 Trier par",
        ("Date récente", "Date ancienne", "Titre A → Z", "Titre Z → A")
    )

    if recherche:
        urls_df = urls_df[urls_df.apply(lambda row: recherche.lower() in (str(row["titre"])+str(row["resume"])+str(row["themes"])).lower(), axis=1)]

    if tri == "Date récente":
        urls_df = urls_df.sort_values("date", ascending=False)
    elif tri == "Date ancienne":
        urls_df = urls_df.sort_values("date", ascending=True)
    elif tri == "Titre A → Z":
        urls_df = urls_df.sort_values("titre", ascending=True)
    elif tri == "Titre Z → A":
        urls_df = urls_df.sort_values("titre", ascending=False)

    st.markdown(f"### 🎬 {len(urls_df)} vidéo(s) trouvée(s)")

    for _, row in urls_df.iterrows():
        video_name = row.get("titre", "Titre inconnu")
        video_date = row.get("date", "Date inconnue")
        url_complet = row["url"]
        resume = row.get("resume", "")
        themes = row.get("themes", "")
        idees = row.get("idees", [])

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

            if themes:
                tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
                for theme in themes.split("|"):
                    theme = theme.strip()
                    if theme:
                        tags_html += f"<a style='background-color: #e1e4e8; padding: 5px 10px; border-radius: 15px; text-decoration: none; color: black; font-size: 14px;' href='?theme={theme}'>{theme}</a>"
                tags_html += "</div>"
                st.markdown(tags_html, unsafe_allow_html=True)

            if idees:
                with st.expander("🎯 Grands moments de la vidéo"):
                    for idee_obj in idees:
                        idee = idee_obj["idee"]
                        start = int(idee_obj["start"])
                        youtube_link = f"https://www.youtube.com/watch?v={youtube_id}&t={start}s"
                        st.markdown(f"- [{idee}]({youtube_link})")

            st.markdown(f"[▶️ Voir sur YouTube]({url_complet})")

        st.markdown("---")
