import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requests
import json
import h5py
from io import BytesIO
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import keras
# Page title
st.set_page_config(page_title='Image Diagnosis Prediction', page_icon='🩺')
st.title('🩺 Image Diagnosis Prediction')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to upload an image and get a diagnosis prediction using a pre-trained model.')

# Sidebar for test and model selection
with st.sidebar:
    st.header('Test Selection')
    test_option = st.selectbox("Choose a test", ["ImmunoComb Peritonite Infecciosa Felina"])
    
    if test_option == "ImmunoComb Peritonite Infecciosa Felina":
        model_url = "./pif/pif_20240607.joblib"  # Replace with the actual URL of your joblib file
        class_url = "./pif/pif_class_indices.json"
        description_url = "./pif/pif_class_descriptions.json"
@st.cache_resource
def load_model(url):
    model = joblib.load(model_url)
    return model

if model_url:
    try:
        st.write("Loading model...")
        model = load_model(model_url)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Main section for image upload and prediction
st.header('Upload an Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if model_url and model:
        st.write("Classifying...")
        def predict_image(image):
            img = load_img(image, target_size=(150, 150))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0
            preds = model.predict(x)
            return preds
        preds = predict_image(uploaded_file)
        # Read the JSON file
        with open(class_url, 'r') as file:
            class_indices = json.load(file)
        prediction_class = next(key for key, value in class_indices.items() if value == np.argmax(preds))
        # Read the JSON file
        with open(description_url, 'r') as file:
            descriptions_indices = json.load(file)
            
        resultados = descriptions_indices[prediction_class]
        st.write(f"Resultado: {resultados}")
    else:
        st.warning("Please upload a model file to make predictions.")
else:
    st.warning("Please upload an image to classify.")
