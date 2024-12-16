#!/bin/bash
# Téléchargement des fichiers nécessaires
curl -O https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/projects/447dd4/DIC.zip
curl -O https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/projects/447dd4/dataset_eval.zip

# Installer unzip
apt-get install unzip

# Dézipper les fichiers DIC
unzip DIC.zip

# Dézipper les fichiers d'évaluation
unzip dataset_eval.zip

# Créer un répertoire pour les fichiers d'évaluation
mkdir dataset_eval

# Déplacer les fichiers JSON dans le répertoire d'évaluation
mv *.json ./dataset_eval/

# Suppression des fichiers zip
rm DIC.zip
rm dataset_eval.zip

# Installation des librairies nécessaires
# pip install -r requirements.txt
