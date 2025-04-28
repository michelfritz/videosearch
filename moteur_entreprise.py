import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  # <-- pour interroger OpenAI proprement

# Charger .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

vectordb = FAISS.load_local(
    "faiss_transcripts",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)


# L'utilisateur pose une question
question = input("Pose ta question : ")

# Chercher les documents les plus proches
docs = vectordb.similarity_search(question, k=5)  # On récupère les 5 blocs les plus pertinents

# Préparer le contexte pour GPT
context = ""
for doc in docs:
    url = doc.metadata.get("url", "URL inconnue")
    context += f"[Source: {url}]\n{doc.page_content}\n\n"

# Construire le prompt
prompt = f"""
Tu es un expert de l'entreprise. Voici des extraits des vidéos de formation :

{context}

Réponds précisément à la question suivante en utilisant uniquement ces extraits. 
Si tu ne sais pas, réponds : "Je n'ai pas trouvé cette information dans notre base actuelle."

Question : {question}
"""

# Appel à GPT-4 Turbo
llm = ChatOpenAI(
    model="gpt-4-0125-preview",
    temperature=0.2,  # plus bas pour des réponses fiables
    openai_api_key=openai_api_key
)

response = llm.invoke(prompt)

# Afficher la réponse
print("\n--- Réponse ---\n")
print(response.content)
