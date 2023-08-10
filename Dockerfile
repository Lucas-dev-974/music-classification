# Utilisez une image de base compatible avec Streamlit et Python
FROM python:3.8-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie des fichiers requis dans le conteneur
COPY . /app/

# Installation des dépendances
RUN pip install streamlit keras tensorflow librosa matplotlib

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "main.py"]
