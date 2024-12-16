# Récupération des token et autres clés
from dotenv import load_dotenv
import os
load_dotenv()
TOKEN_HF = os.getenv("TOKEN_HF")


### Nettoyer le texte des documents

def nettoyage_texte(texte: str) -> str:
    # Correction des mots avec trait d'union interrompus par la nouvelle ligne
    texte = re.sub(r'(\w+)-\n(\w+)', r'\1\2', texte)

    # Supprimer des motifs et des caractères indésirables spécifiques
    patterns_indesirables = [
        "  —", "——————————", "—————————", "—————",
        r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
    ]
    for pattern in patterns_indesirables:
        texte = re.sub(pattern, "", texte)

    # Corriger les mots avec trait d'union mal espacés
    texte = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', texte)
    texte = re.sub(r'^[\*\-\u2022]', '', texte)
    
    return texte

### Rechercher les metadata pour chacun des documents

import re

def extraction_metadata(texte: str) -> (str, str, int):
    """
    Recherche du Nom du produit, code ISIN, et Niveau de risque
    """
    produit, isin, niveau_risque = '', '', -1

    # Niveau de risque
    pattern_risque = r"(?:(?:niveau|classe|catégorie)\sde\srisque(?:\sde\sce\scompartiment\sest\sde)?|indicateur\sde\srisque\sde\sniveau|(?:est\sclassé(?:\sdans\sla)?|appartient\sà\sla)\scat[ée]gorie)\s(\d+)"
    recherche_risque = re.search(pattern_risque, texte)
    niveau_risque = int(recherche_risque.group(1)) if recherche_risque else -1

    # Code ISIN (format: 2 lettres, 1 nombre, 9 caractères alphanumeriques)
    recherche_isin = re.search(r"\b[a-zA-Z]{2}[0-9]{1}[0-9A-Z]{9}\b", texte)
    isin = recherche_isin.group() if recherche_isin else ''

    # Nom du produit, dans la phrase contenant ISIN
    if isin:
        phrases = re.split('(?<=[.!?])\s+', texte)
        for phrase in phrases:
            if isin in phrase:
                produit = phrase.strip().replace('\n', ' ')
                break

    return produit, isin, niveau_risque

# Lecture des documents présents dans le répoertoire /DIC
from langchain.schema import Document
import os
import pdfplumber
import fitz

repertoire = "./DIC"   # Répertoire où sont stockés les documents PDF à charger
docs = []   # Initialisation de la liste des documents pdf chargés

# Parcourir tous les fichiers du répertoire
for nom_fichier in os.listdir(repertoire):
    if nom_fichier.endswith('.pdf'):
        chemin_fichier = os.path.join(repertoire, nom_fichier)
        
        pdf = fitz.open(chemin_fichier)
        texte_nettoye = ""
        for page in pdf:
            texte = page.get_text()
            if texte:
                texte = texte.encode('utf-8').decode('utf-8')
# Nettoyer le texte récupéré
                texte = nettoyage_texte(texte)
                texte_nettoye += "\n" + texte   # Concaténation des différentes pages d'un même PDF
        texte_nettoye = texte_nettoye.lstrip('\n')   # Retirer le saut de ligne en début de document
        
# Extraire les metadata
        produit, isin, niveau_risque = extraction_metadata(texte_nettoye)
        metadata = {
            'source': nom_fichier,
#            'produit': produit,
#            'isin': isin,
#            'niveau_risque': niveau_risque,
            # Ajoutez d'autres métadonnées ici si nécessaire
        }
        doc = Document(page_content=texte_nettoye, metadata=metadata)
        docs.append(doc)  # Ajouter le document à la liste des documents

# Afficher le nombre de pages de documents chargées
print(f"{len(docs)} PDF chargées.")

print(docs[4].page_content)

# Découper les documents (chunks)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# On défini la taille du split et le recouvrement
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,            # Taille plus grande pour maintenir plus de contexte
    chunk_overlap=100,          # Chevauchement pour maintenir la continuité
    length_function=len,
    separators=[
        "\n\n",               # D'abord essayer de séparer par paragraphes
        "\n",                 # Puis par lignes
        ". ",                 # Puis par phrases
        ""                    # Si rien d'autre ne fonctionne
    ]
)

# On découpe alors les documents
chunked_docs = splitter.split_documents(docs)

# Afficher le nombre de chunk de documents
print(f"{len(chunked_docs)} chunk")

# Afficher les metadonnées d'un chunk
print(chunked_docs[4].metadata)
print("/n")
print(chunked_docs[4].page_content)


# Charger le modele d'embedding

from langchain_huggingface import HuggingFaceEmbeddings
model_embed = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# Générer et sauvegarder les embeddings dans le vectorstore (ChromaDB ou Faiss par ex.)

from langchain.vectorstores import FAISS

texts = [doc.page_content for doc in chunked_docs]
metadatas = [doc.metadata for doc in chunked_docs]

db = FAISS.from_texts(
    texts=texts,
    embedding=model_embed,
    metadatas=metadatas
    )

# Chargement du modèle de LLM à utiliser

from huggingface_hub import login

hf_token = TOKEN_HF  # Token HuggingFace stocké dans le .env
login(token=hf_token)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Chargement du modèle
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)


# Initialisation de la chaine LangChain du LLM (Prompt + modèle)

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    top_p=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=512,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<role>
Vous êtes un assistant d'une équipe du département ALM (gestion des actifs / passifs) dans une entreprise en assurance vie.
Votre rôle est de répondre à la question entre <question></question> en suivant le contexte entre <contexte></contexte> afin d'assister l'équipe dans le choix des placements à effectuer.
Le contexte entre <contexte></contexte> contient des informations extraites de document d'informations clés (DIC). Le DIC est un document harmonisé au niveau européen \n
qui permet de retrouver les informations essentielles sur un placement, sa nature et ses caractéristiques principales.
Pour Répondre à la question entre <question></question> tu dois absolument suivre les instructions entre <instructions></instructions>.
</role>

<instructions>
La REPONSE doit être concise et rédigée en Français. 
La REPONSE doit être basée sur le contexte entre <contexte></contexte>.
Si le contexte entre <contexte></contexte> ne permet pas de répondre à la question entre <question></question>, réponds "Je ne peux pas répondre à votre demande".
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

llm_chain = prompt | llm | StrOutputParser()

# Construction du Retriever = chercher passages de documents / à la similarité vectorielle

from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

rag_chain = RunnablePassthrough.assign(
    context=lambda x: retriever.get_relevant_documents(x["question"])

) | llm_chain

question = "Quels critères sont pris en compte dans les décisions d'investissement du FCP ?"

# Test : Question en direct sans passer par le RAG
reponse = llm_chain.invoke({"context":"", "question": question})

reponse_llm = reponse.split(":::")[-1].strip()
print(reponse_llm)

# Question en passant par le RAG
def llm_rag_answer (question):
    reponse = rag_chain.invoke({"question" : question})
    sources = sources_reponse(reponse)
    reponse_llm = reponse.split(":::")[-1].strip()
    return(reponse_llm, sources)

# Extraction des sources desquelles sont issues la réponse du llm
def sources_reponse(reponse):
    # Extraction de la source à partir de la chaîne de caractères dans le contexte
    matches = re.findall(r"'source': '(.*?)'", reponse)
    if matches:
        return matches
    else:
        return "Source non trouvée"

reponse_llm, sources = llm_rag_answer (question)
print(reponse_llm)

# Test de l'appel du retriever en direct
test_retriever = retriever.invoke("Comment obtenir un exemplaire papier de la politique de rémunération ?")
print(test_retriever, "\n")
print("-"*10)

# Et de recherche des sources dans les metadata
def retrieve_with_metadata(reponses):
    return [reponse.metadata['source'] for reponse in reponses]

retrieve_with_metadata(test_retriever)

# Evaluer le modèle via un F1Score BERT

## Préparation du modèle pour le F1Score

# Première étape du travail : récupérer les données en lien dans chacun des fichiers JSON
# Il faudra ensuite passer au modèle dans une BD Chroma dédiée l'ensemble des chunks issus du fichier corpus
# Ensuite on boucle sur le fichier Query et on envoit au modèle la query pour récupérer une réponse
# On stocke le couple query / réponse dans un tableau (DataFrame ? à creuser)
# Enfin on aura les réponse attendues et les réponses prédites, on les compare en les donnant au modèle BERT F1Score
# Ci-dessous j'ai pris un exemple : b9997872-bff0-4ee0-b181-caf6b9c2560b

### Lecture des documents au format JSON servant a évaluer le modèle

import json

# Chemin des fichiers d'évaluation
relevant_docs_file_path = "./dataset_eval/relevant_docs.json"
corpus_file_path = "./dataset_eval/corpus.json"
queries_file_path = "./dataset_eval/queries.json"
answers_file_path = "./dataset_eval/answers.json"
errors_file_path = "./dataset_eval/errors.json"

# Chargement du fichier relevant_docs : liste des documents pertinents pour chaque requête.
with open(relevant_docs_file_path, 'r', encoding='utf-8') as file:
    relevant_docs = json.load(file)

# Chargement du fichier corpus = ensemble de corpus, c'est-à-dire tous les chunks que composent les documents.
with open(corpus_file_path, 'r', encoding='utf-8') as file:
    corpus = json.load(file)

# Chargement du fichier queries : requêtes formulées sur chaque corpus.
with open(queries_file_path, 'r', encoding='utf-8') as file:
    queries = json.load(file)

# Chargement du fichier answers : la réponse attendue à chaque requête.
with open(answers_file_path, 'r', encoding='utf-8') as file:
    answers = json.load(file)

# Chargement du fichier errors : erreurs éventuelles lors de l'inférence
with open(errors_file_path, 'r', encoding='utf-8') as file:
    errors = json.load(file)

### Constitution d'un fichier JSON de correspondance entre les réponses attendues et les réponses données par le RAG

# Chemin du fichier JSON dans lequel sauvegarder les correspondances entre questions, réponses du RAG et réponses attendues par la base d'évaluation
reponse_eval_rag_file_path = "./dataset_eval/answers_rag_eval.json"
resultat = {}
resultats = []
sources = []

# Créer ou vider le fichier de sortie
with open(reponse_eval_rag_file_path, 'w', encoding='utf-8') as file:
    file.write('')  # Vider le fichier s'il existe déjà

for i, (uuid, question) in enumerate(queries.items(),1):
    reponse_rag, sources = llm_rag_answer(question)
    # Créer l'objet résultat
    resultat = {
        "id_question": uuid,
        "question": question,
        "reponse_rag": reponse_rag,
        "reponse_attendue": answers.get(uuid, "Réponse attendue non trouvée"),
        "sources" : sources
    }
    resultats.append(resultat)
    if i == 10:
        break
    
# Écrire le résultat dans le fichier
with open(reponse_eval_rag_file_path, 'w', encoding='utf-8') as file:
    json.dump(resultats, file, ensure_ascii=False)
    

import pandas as pd

# Charger l'ensemble du fichiers JSON des résultats dans un DF
with open(reponse_eval_rag_file_path, 'r') as file:
    data = json.load(file)
df_eval_rag = pd.DataFrame(data)
df_eval_rag.set_index('id_question', inplace=True)

# Extraire la colonne reponse_rag
cands = df_eval_rag['reponse_rag'].tolist()

# Extraire la colonne reponse_attendue
refs = df_eval_rag['reponse_attendue'].tolist()

# BERT SCORE une fois les réponses obtenues de la part du modèle

# Evaluation BERT pour le modèle complet

"""
# Nécessaire si les réponses sont stoquées dans un fichier
# Chargement fichiers CSV
import pandas as pd

df_cands = pd.read_csv('./BERTScore/reponses_rag.csv')
cands = df_cands['0'].tolist()

df_refs = pd.read_csv('./BERTScore/reponses_attendues.csv')
refs = df_refs['0'].tolist()

print(cands)
print(refs)
"""

from bert_score import score

# Calcul du F1 score pour chacune des Réponses données
P, R, F1 = score(cands, refs, lang='fr', verbose=True)

# Le score de chacun des tests réalisés successivement
print(F1)

# Pour le score du modèle complet :
print(f"System level F1 score: {F1.mean():.3f}")

### Affichage graphique du F1Score pour une meilleure compréhension

import logging
import transformers

import matplotlib.pyplot as plt
from matplotlib import rcParams

from bert_score import plot_example

# hide the loading messages
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

rcParams["xtick.major.size"] = 0
rcParams["xtick.minor.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["ytick.minor.size"] = 0

rcParams["axes.labelsize"] = "large"
rcParams["axes.axisbelow"] = True
rcParams["axes.grid"] = True

# Affichage global
plt.hist(F1, bins=20)
plt.xlabel("score")
plt.ylabel("counts")
plt.show()

# Plot pour un seul chunk
plot_example(cands[8], refs[8], lang="fr")
plot_example(cands[8], refs[8], lang="fr", rescale_with_baseline=True)   # Apply rescaling to adjust the similarity distribution to be more distinguishable.

# Evaluation de la partie LLM seule : réponse à la querie (sans passer par le RAG) et comparaison à la réponse du DATASET d'éval

llm_chain.invoke({"context":"", "question": question})   #Question en direct sans passer par le RAG

from bert_score import score

# Calcul du F1 score pour chacune des Réponses données
P, R, F1 = score(cands, refs, lang='fr', verbose=True)

# Le score de chacun des tests réalisés successivement
print(F1)

# Pour le score du modèle complet :
print(f"System level F1 score: {F1.mean():.3f}")



# Reste à faire

# Sourcer en utilisant les metadata du RAG

# Mettre en place une mémoire, pas trop longue, pourquoi pas en utilisant un résumé de l'historique de l'échange
# ou en limitant aux 3 dernières inférences

# Prévoir les évaluations de chacune des parties du modèle : Retriever, LLM lui même (formulation)...

# Sauvegarder en local ma base de données vectorielle ChromaDB
