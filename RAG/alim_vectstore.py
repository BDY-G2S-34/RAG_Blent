from langchain.schema import Document
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st  # Importe streamlit pour créer l'application web
from typing import Tuple
import fitz
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm


# Récupération des token et autres clés
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Nettoyer le texte des documents en entrée, avant ce pouvoir réaliser
# les chunks et embeddings
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

# Rechercher les metadata pour chacun des documents


def extraction_metadata(texte: str) -> Tuple[str, str, int]:
    """
    Recherche du Nom du produit, code ISIN, et Niveau de risque
    """
    produit, isin, niveau_risque = '', '', -1

    # Niveau de risque
    pattern_risque = (
        r"(?:(?:niveau|classe|catégorie)\sde\srisque"
        r"(?:\sde\sce\scompartiment\sest\sde)?|"
        r"indicateur\sde\srisque\sde\sniveau|(?:est\sclassé(?:\sdans\sla)?|"
        r"appartient\sà\sla)\scat[ée]gorie)\s(\d+)"
    )
    recherche_risque = re.search(pattern_risque, texte)
    niveau_risque = int(recherche_risque.group(1)) if recherche_risque else -1

    # Code ISIN (format: 2 lettres, 1 nombre, 9 caractères alphanumeriques)
    recherche_isin = re.search(r"\b[a-zA-Z]{2}[0-9]{1}[0-9A-Z]{9}\b", texte)
    isin = recherche_isin.group() if recherche_isin else ''

    # Nom du produit, dans la phrase contenant ISIN
    if isin:
        phrases = re.split(r'(?<=[.!?])\s+', texte)
        for phrase in phrases:
            if isin in phrase:
                produit = phrase.strip().replace('\n', ' ')
                break

    return produit, isin, niveau_risque

workdirectory = os.getcwd() # répertoire de travail actuel
repertoire = os.path.join(workdirectory, "DIC")   # Répertoire où sont stockés les documents PDF à charger
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
# Concaténation des différentes pages d'un même PDF
                texte_nettoye += "\n" + texte
# Retirer le saut de ligne en début de document
        texte_nettoye = texte_nettoye.lstrip('\n')
# Extraire les metadata
        produit, isin, niveau_risque = extraction_metadata(texte_nettoye)
        metadata = {
            '<source>': nom_fichier,
            '<produit>': produit,
            '<isin>': isin,
            '<niveau_risque>': niveau_risque,
            # Ajoutez d'autres métadonnées ici si nécessaire
        }
        doc = Document(page_content=texte_nettoye, metadata=metadata)
        docs.append(doc)  # Ajouter le document à la liste des documents

# Découper les documents (chunks)
# On défini la taille du split et le recouvrement
splitter = RecursiveCharacterTextSplitter(
    # Taille plus grande pour maintenir plus de contexte
    chunk_size=512,
    # Chevauchement pour maintenir la continuité
    chunk_overlap=100,
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


# Création d'un index dans le cloud (servless)
# La dimension doit être cohérente avec celle du modèle d'embedding
pc = Pinecone(api_key=PINECONE_API_KEY)

nom_index = "bvragblent"

if nom_index not in pc.list_indexes():
    pc.create_index(
        name=nom_index,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Instanciation d'un objet Index
index = pc.Index(nom_index)

from langchain_huggingface import HuggingFaceEmbeddings
model_embed = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",    #"intfloat/multilingual-e5-large",
    encode_kwargs={'normalize_embeddings': True}
)

# Préparer les données
texts = [doc.page_content for doc in chunked_docs]
metadatas = [doc.metadata for doc in chunked_docs]

# Embedding des donées et insertion dans l'index Pinecone
batch_size = 100
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]
    batch_metadata = metadatas[i:i + batch_size]
    
    embeddings = model_embed.embed_documents(batch_texts)
    
    vectors = []
    for j, (text, embedding) in enumerate(zip(batch_texts, embeddings)):
        metadata = batch_metadata[j]
        metadata["page_content"] = text  # Ajout du contenu dans les métadonnées
        vectors.append((
            f"doc_{i+j}",
            embedding,
            metadata
        ))
    
    index.upsert(vectors=vectors)
