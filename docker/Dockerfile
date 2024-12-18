# Utiliser une image de base officielle de Python
FROM python:3.10.5-slim

# Définir les variables d'environnement
ENV TOKEN_HF = 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
ENV URL_OLLAMA = https://d172-207-102-87-207.ngrok-free.app 

# Installer Git
RUN apt-get update && apt-get install -y git

# Définir le répertoire de travail dans le conteneur
WORKDIR ./

# Cloner le dépôt GitHub
RUN git clone https://github.com/BDY-G2S-34/RAG_Blent.git .

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r ./RAG_Blent/requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Spécifier la commande à exécuter lorsque le conteneur démarre
CMD ["python -m", "streamlit", "run", "./IHM/app.py", "--server.port=8501", "--server.address=0.0.0.0"]]