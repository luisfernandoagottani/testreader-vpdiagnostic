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
st.set_page_config(page_title='Predição de Diagnóstico por Imagem', page_icon='🩺')
st.title('🩺 Predição de Diagnóstico por Imagem')

st.info('Este aplicativo permite que os usuários enviem uma imagem e obtenham uma previsão de diagnóstico usando um modelo pré-treinado...')

# Sidebar for test and model selection

st.header('Seleção de teste diagnóstico')
test_option = st.selectbox("Escolha um teste:", ["ImmunoComb Peritonite Infecciosa Felina (PIF)"])
    
if test_option == "ImmunoComb Peritonite Infecciosa Felina (PIF)":
    model_url = "./pif/pif_20240607.joblib"  # Replace with the actual URL of your joblib file
    class_url = "./pif/pif_class_indices.json"
    description_url = "./pif/pif_class_descriptions.json"
    test_info = "Teste Dot-Elisa que determina no soro, plasma de gatos o nível de anticorpo IgG contra Coronavírus Felino (FCoV). Gatos com Peritonite Infecciosa Felina (PIF) contém altos níveis de anticorpo antiCoronavírus Felino. Também pode ser utilizado líquido peritonial como amostra. O resultado negativo é útil para afastar um diagnóstico da PIF."

st.info(test_info)

@st.cache_resource
def load_model(url):
    model = joblib.load(model_url)
    return model

if model_url:
    try:
        st.write("Carregando modelo...")
        model = load_model(model_url)
        st.success("Modelo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")

# Main section for image upload and prediction
st.header('Carregar imagem')
st.image('exemplo_teste.png', caption='Exemplo de fotografia', width=300)
uploaded_file = st.file_uploader("Selecione uma imagem, de preferência com fundo branco e apenas um teste por vez...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem carregada.', width=200)

    if model_url and model:
        st.write("Classificando...")
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
        if preds.max() > 0.8:
            st.write(f"Resultado: {resultados}")
            st.write(f"Confiança: {round(preds.max()*100,2)} %")
        else:
            st.write("Pouca confiança na previsão. Se for possível, tente usar fundo branco e apenas um teste por vez.")
    else:
        st.warning("Carregar uma imagem para realizar a predição.")
else:
    st.warning("Carregar uma imagem para realizar a predição.")
