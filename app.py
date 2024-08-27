import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Charger le modèle TensorFlow .h5
model = tf.keras.models.load_model("mon_modele.h5")

# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = image.resize((224, 224))  # Redimensionner l'image à la taille d'entrée du modèle
    image = np.array(image)  # Convertir l'image en tableau numpy
    image = image / 255.0  # Normaliser l'image
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    return image

# Fonction pour faire une prédiction
def make_prediction(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]  # Supposant que le modèle retourne une seule prédiction

# Titre de l'application
st.title("Détection de Dust avec Deep Learning")

# Uploader pour charger une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

# Si un fichier est téléchargé
if uploaded_file is not None:
    # Ouvrir l'image avec PIL
    image = Image.open(uploaded_file)

    # Afficher l'image téléchargée
    st.image(image, caption="Image chargée avec succès", use_column_width=True)

    # Faire une prédiction
    prediction = make_prediction(image, model)

    # Afficher le résultat
    if prediction < 0.5:
        st.write("Résultat : Pas de dust détecté.")
    else:
        st.write("Résultat : Dust détecté.")
else:
    st.write("Veuillez télécharger une image pour obtenir une prédiction.")
