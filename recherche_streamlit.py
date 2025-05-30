from openai import OpenAI
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
import chardet
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="Base de connaissance A LA LUCARNE", layout="wide")

# 🎨 Logo
st.image("logo_lucarne.png", width=180)
st.markdown("# 📚 Base de connaissance A LA LUCARNE")


# 🔐 Clé API OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")


# 📂 Dossier newsletters
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



# 🔥 Détection encodage
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

# 📚 Charger les données blindées
@st.cache_data
def charger_donnees():
    df = pd.read_csv("blocs_fusionnes.csv")
    with open("vecteurs.pkl", "rb") as f:
        vecteurs = pickle.load(f)
    return df, vecteurs

@st.cache_data
def charger_urls_et_idees_themes():
    urls = pd.read_csv("urls.csv", encoding=detect_encoding("urls.csv"))
    urls["titre"] = urls["titre"].fillna("Titre inconnu")
    urls["date"] = urls["date"].fillna("Date inconnue")
    urls["resume"] = urls["resume"].fillna("")

    idees = pd.read_csv("idees.csv", encoding=detect_encoding("idees.csv"))
    idees["idees"] = idees["idees"].fillna("")

    idees_v2 = pd.read_csv("idees_v2.csv", encoding=detect_encoding("idees_v2.csv"))

    themes = pd.read_csv("themes.csv", encoding=detect_encoding("themes.csv"))
    themes["themes"] = themes["themes"].fillna("")

    mesthemes = pd.read_csv("mesthemes.csv", encoding=detect_encoding("mesthemes.csv"))
    mesthemes_list = mesthemes["themes"].dropna().tolist()

    df = pd.merge(urls, idees, on="fichier", how="left")
    df = pd.merge(df, themes, on="fichier", how="left")
    return df, idees_v2, themes, mesthemes_list

# 🔎 Embedding OpenAI
def embed_openai(query):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small",
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)



# 🔥 Recherche vectorielle
def rechercher_similaires(vecteur_query, vecteurs, top_k=5, seuil=0.3):
    similarities = np.dot(vecteurs, vecteur_query)
    indices = np.where(similarities >= seuil)[0]
    top_indices = indices[np.argsort(similarities[indices])[::-1][:top_k]]
    return top_indices, similarities[top_indices]

# 🛠 Interface
df, vecteurs = charger_donnees()
urls_df, idees_v2_df, themes_df, mesthemes_list = charger_urls_et_idees_themes()

# 🔖 Préparer tous les thèmes
all_themes = set()
for theme_list in themes_df["themes"].dropna():
    for theme in theme_list.split("|"):
        theme = theme.strip()
        if theme:
            all_themes.add(theme)

# 🧠 Gérer session
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = ""

if "reset_search" not in st.session_state:
    st.session_state.reset_search = False

menu = st.sidebar.radio("Navigation", ["🔍 Recherche", "🎥 Toutes les vidéos", "🧠 Moteur intelligent"])


if menu == "🔍 Recherche":
    col1, col2 = st.columns([3,1])

    # Réinitialiser si besoin
    if st.session_state.reset_search:
        if "search_query" in st.session_state:
            del st.session_state["search_query"]
        st.session_state.reset_search = False

    # Champ de recherche
    with col1:
        st.text_input("🔍 Que veux-tu savoir ?", key="search_query")
    
    # Bouton Réinitialiser
    with col2:
        if st.button("🔄 Réinitialiser"):
            st.session_state.selected_theme = ""
            st.session_state.reset_search = True
            st.experimental_rerun()

    seuil = st.slider("🌟 Exigence des résultats", 0.1, 0.9, 0.5, 0.05)

    # 🌟 Mes Thèmes personnalisés
    with st.expander("✨ Thèmes", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(mesthemes_list)):
            if cols[i % 4].button(theme, key=f"mestheme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.reset_search = True
                st.experimental_rerun()

    # 🌟 Tous les Thèmes
    with st.expander("🏷️ Tags", expanded=False):
        cols = st.columns(4)
        for i, theme in enumerate(sorted(all_themes)):
            if cols[i % 4].button(theme, key=f"theme_{theme}"):
                st.session_state.selected_theme = theme
                st.session_state.reset_search = True
                st.experimental_rerun()

    # Définir la requête
    query = st.session_state.get("search_query", "").strip() or st.session_state.get("selected_theme", "").strip()

    if query:
        with st.spinner("🔍 Recherche en cours..."):
            vecteur_query = embed_openai(query)
            indices, scores = rechercher_similaires(vecteur_query, vecteurs, seuil=seuil)

        if len(indices) == 0:
            st.warning("Aucun résultat trouvé.")
        else:
            st.markdown("### 🌟 Résultats pertinents :")
            for idx, score in zip(indices, scores):
                bloc = df.iloc[idx]
                url_complet = bloc["url"]
                youtube_id = ""
                if "watch?v=" in url_complet:
                    youtube_id = url_complet.split("watch?v=")[-1]
                elif "youtu.be/" in url_complet:
                    youtube_id = url_complet.split("youtu.be/")[-1]

                start_time = int(float(bloc["start"]))
                embed_url = f"https://www.youtube.com/embed/{youtube_id}?start={start_time}&autoplay=0"

                with st.expander(f"⏱️ {start_time}s — 💬 {bloc['text'][:60]}... (score: {score:.2f})"):
                    st.markdown(f"**Texte complet :** {bloc['text']}")
                    if youtube_id:
                        st.components.v1.iframe(embed_url, height=315)

elif menu == "🎥 Toutes les vidéos":
    st.header("📚 Liste des vidéos disponibles")
    recherche = st.text_input("🔍 Recherche par titre, résumé, idée ou thème", key="video_search")

    tri = st.selectbox("📜 Trier par", ("Date récente", "Date ancienne", "Titre A → Z", "Titre Z → A"))

    if recherche:
        urls_df = urls_df[urls_df.apply(lambda row: recherche.lower() in (str(row["titre"])+str(row["resume"])+str(row["idees"])+str(row["themes"])).lower(), axis=1)]

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
        url_complet = row.get("url", "")
        resume = row.get("resume", "")
        idees = row.get("idees", "")
        themes = row.get("themes", "")
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

 # Bouton Newsletter ici
            if fichier_nom:
                if st.button("📰 Voir Newsletter", key=f"newsletter_{fichier_nom}"):
                    newsletter_contenu = charger_newsletter_html(fichier_nom)
                    if newsletter_contenu:
                        with st.expander("📬 Newsletter liée à cette vidéo"):
                            st.markdown(newsletter_contenu, unsafe_allow_html=True)
                            bouton_telecharger_newsletter(fichier_nom, newsletter_contenu)
                    else:
                        st.warning("❌ Pas de newsletter disponible pour cette vidéo.")



            if themes:
                tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
                for theme in themes.split("|"):
                    theme = theme.strip()
                    if theme:
                        tags_html += f"<span style='background-color: #D0E8FF; padding: 6px 12px; border-radius: 20px;'>{theme}</span>"
                tags_html += "</div>"
                st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            if idees:
                with st.expander("🌟 Sujets de la vidéo"):
                    for idee in idees.split("|"):
                        idee = idee.strip()
                        if idee and youtube_id:
                            st.markdown(f"- [{idee}](https://www.youtube.com/watch?v={youtube_id}&t=0s)")
                        elif idee:
                            st.markdown(f"- {idee}")

            if fichier_nom:
                with st.expander("🕒 Moments de la vidéo"):
                    idees_v2_video = idees_v2_df[idees_v2_df["fichier"] == fichier_nom]
                    for _, idee_row in idees_v2_video.iterrows():
                        idee_text = idee_row.get("idee", "")
                        start_time = int(float(idee_row.get("start", 0)))
                        if idee_text and youtube_id:
                            st.markdown(f"- [{idee_text}](https://www.youtube.com/watch?v={youtube_id}&t={start_time}s)")
                        elif idee_text:
                            st.markdown(f"- {idee_text}")

        st.markdown("---")

elif menu == "🧠 Moteur intelligent":
    st.header("🧠 Assistant IA basé sur vos formations vidéos")
    
    user_question = st.text_input("Pose ta question :", key="user_question")
    
    if user_question:
        with st.spinner("Recherche intelligente en cours..."):
            # Charger FAISS
            vectordb = FAISS.load_local(
                "faiss_transcripts",
                OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")),
                allow_dangerous_deserialization=True
            )

            # Recherche dans FAISS
            docs = vectordb.similarity_search(user_question, k=5)

            # Contexte pour GPT
            context = ""
            for doc in docs:
                url = doc.metadata.get("url", "URL inconnue")
                context += f"[Source: {url}]\n{doc.page_content}\n\n"

            # Construire prompt
            prompt = f"""
Tu es un expert de notre entreprise. Voici des extraits de nos formations :

{context}

Réponds précisément à la question suivante en utilisant uniquement ces extraits.
Si aucune information n'existe, réponds : "Je n'ai pas trouvé cette information dans notre base actuelle."

Question : {user_question}
"""

            # Appel à GPT-4 Turbo
            llm = ChatOpenAI(
                model="gpt-4-0125-preview",
                temperature=0.2,
                openai_api_key=openai.api_key
            )
            response = llm.invoke(prompt)

            # Afficher la réponse
            st.success(response.content)

