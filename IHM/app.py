from langchain.schema import Document
import os
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st  # Importe streamlit pour créer l'application web
from ollama import Client
from langchain_chroma import Chroma
from typing import Tuple
from langchain_huggingface import HuggingFaceEmbeddings
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


# Charger le modele d'embedding
model_embed = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
persist_directory = "./db"
# Chargement de la base Chroma
db = Chroma(persist_directory = persist_directory, embedding_function = model_embed)

prompt_template = """
<role>
Vous êtes un assistant d'une équipe du département ALM (gestion des actifs /
passifs) dans une entreprise en assurance vie.
Votre rôle est de répondre à la question entre <question></question> en
suivant le contexte entre <contexte></contexte> afin d'assister l'équipe dans
le choix des placements à effectuer.
Le contexte entre <contexte></contexte> contient des informations extraites de
document d'informations clés (DIC). Le DIC est un document harmonisé au niveau
européen \n
qui permet de retrouver les informations essentielles sur un placement, sa
nature et ses caractéristiques principales.
Pour Répondre à la question entre <question></question> tu dois absolument
suivre les instructions entre <instructions></instructions>.
</role>

<instructions>
La REPONSE doit être concise et rédigée en Français.
La REPONSE doit être basée sur le contexte entre <contexte></contexte>.
Si le contexte entre <contexte></contexte> ne permet pas de répondre à la
question entre <question></question>, réponds "Je ne peux pas répondre à votre
demande".
</instructions>

<contexte>
{context}
</contexte>

<question>
{question}
</question>

REPONSE :::
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Construction du Retriever = chercher passages de documents
# / à la similarité vectorielle
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

llm = OllamaLLM(
    model='mistral',
    base_url='https://7393-174-164-27-23.ngrok-free.app'   # or your Ollama server URL
)

llm_chain = prompt | llm | StrOutputParser()

rag_chain = RunnablePassthrough.assign(
    context=lambda x: retriever.get_relevant_documents(x["question"])
) | llm_chain

# question = "Quels critères sont pris en compte dans les décisions
# d'investissement du FCP ?"


# Question en passant par le RAG
def llm_rag_answer(question):
    reponse = rag_chain.invoke({"question": question})
    sources = sources_reponse(reponse)
    reponse_llm = reponse.split(":::")[-1].strip()
    return (reponse, reponse_llm, sources)


# Extraction des sources desquelles sont issues la réponse du llm
def sources_reponse(reponse):
    # Extraction de la source à partir de la chaîne de caractères
    # dans le contexte
    matches = re.findall(r"'source': '(.*?)'", reponse)
    if matches:
        return matches
    else:
        return "Source non trouvée"

# Reste à faire

# Sourcer en utilisant les metadata du RAG

# Mettre en place une mémoire, pas trop longue, pourquoi pas en utilisant
# un résumé de l'historique de l'échange ou en limitant
# aux 3 dernières inférences

# Prévoir les évaluations de chacune des parties du modèle :
# Retriever, LLM lui même (formulation)...

# Sauvegarder en local ma base de données vectorielle ChromaDB


###########################
# Création de l'appli Web


# TODO : Remplacer par l'URL affichée dans Ngrok
client = Client(host='https://7393-174-164-27-23.ngrok-free.app')

# Définit le titre de l'application web
st.title("Chatbot Ollama")

# Initialise l'historique des messages s'il n'existe pas dans
# l'état de la session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialise le modèle sélectionné s'il n'existe pas dans l'état de la session
if "model" not in st.session_state:
    st.session_state["model"] = ""

# Modèle utilisé
st.session_state["model"] = "mistral:latest"


# Définit une fonction générateur pour produire la réponse
# du modèle par morceaux
def model_res_generator():
    # Démarre un flux de discussion avec le modèle sélectionné
    # et l'historique des messages
    stream, reponse_llm, sources = llm_rag_answer(prompt)

    # Produit chaque morceau de la réponse du modèle
    for chunk in stream:
        yield chunk


# Affiche l'historique de la discussion des sessions précédentes
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Récupère l'entrée de l'utilisateur à partir de la zone
# de saisie de discussion
if prompt := st.chat_input("Entrez votre message ici..."):

    # Ajoute le message de l'utilisateur à l'historique
    st.session_state["messages"].append({"role": "user",
                                         "content": prompt})

    # Affiche le message de l'utilisateur dans la discussion
    with st.chat_message("user"):
        st.markdown(prompt)

    # Affiche la réponse de l'assistant dans la discussion
    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant",
                                             "content": message})
