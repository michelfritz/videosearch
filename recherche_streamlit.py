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

# (le reste du script ne change pas)
