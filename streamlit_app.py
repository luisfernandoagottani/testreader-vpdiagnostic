import streamlit as st
import numpy as np
from PIL import Image
from PIL import ImageOps
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

st.set_page_config(page_title='Leitura do Diagnóstico por Imagem', page_icon='🩺')
st.image('LOGO_VP_V_ORIGINAL.png', use_column_width=True)
st.title('🩺 Leitura do Diagnóstico por Imagem')

st.info('Este aplicativo permite que os usuários enviem uma imagem e obtenham uma leitura do teste diagnóstico usando um modelo pré-treinado...')

# Sidebar for test and model selection

st.header('Seleção de teste diagnóstico')
test_option = st.selectbox("Escolha um teste:", ["ImmunoComb Peritonite Infecciosa Felina (PIF) Coronavírus Felino (FCoV) IgG","ImmunoComb Ehrlichia Canis IgG","ImmunoComb Leptospira canina IgG"])

if test_option == "ImmunoComb Peritonite Infecciosa Felina (PIF) Coronavírus Felino (FCoV) IgG":
    model_url = "./pif/cnn_hvs_20240620.joblib"  # Replace with the actual URL of your joblib file
    class_url = "./pif/pif_class_indices.json"
    description_url = "./pif/pif_class_descriptions.json"
    test_info = "Teste Dot-Elisa que determina no soro, plasma de gatos o nível de anticorpo IgG contra Coronavírus Felino (FCoV). Gatos com Peritonite Infecciosa Felina (PIF) contém altos níveis de anticorpo antiCoronavírus Felino. Também pode ser utilizado líquido peritonial como amostra. O resultado negativo é útil para afastar um diagnóstico da PIF."
elif test_option == "ImmunoComb Ehrlichia Canis IgG":
    model_url = "./pif/cnn_hvs_20240620.joblib"  # Replace with the actual URL of your joblib file
    class_url = "./pif/pif_class_indices.json"
    description_url = "./pif/pif_class_descriptions.json"
    test_info = "Teste Dot-Elisa que detecta Ehrlichia canis a partir de 20 dias após a infecção. Semi-quantitativo, informa ao Clínico se a doença está regredindo. Fácil de usar – Teste de amostras individuais."
elif test_option == "ImmunoComb Leptospira canina IgG":
    model_url = "./pif/cnn_hvs_20240620.joblib"  # Replace with the actual URL of your joblib file
    class_url = "./pif/pif_class_indices.json"
    description_url = "./pif/pif_class_descriptions.json"
    test_info = "O kit Canine Leptospira foi projetado para determinar em soros de caninos, títulos de anticorpos séricos de diferentes sorovares patogênicos de Leptospira interrogans, como L. icterohaemorrhagiae (copenhageni e RGA), L. canicola, L. pomona e L. grippotyphosa, ligando anticorpos para as variantes mais encontradas em cães."
st.info(test_info)

@st.cache_resource
def load_model(url):
    model = joblib.load(model_url)
    return model

if model_url:
    try:
        model = load_model(model_url)
    except Exception as e:
        st.write("Erro. O modelo não está disponível.")

# Main section for image upload and prediction
st.header('Para realizar a leitura do teste diagnóstico por imagem, carregue a imagem conforme exemplo.')
st.image('exemplo_teste.png', caption='Exemplo de Imagem', width=300)
uploaded_file = st.file_uploader("Selecione uma imagem, de preferência com fundo branco e apenas um teste por vez...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    def fix_orientation(image):
        try:
            # Fix orientation if needed
            image = Image.open(image).convert("RGB")
            image = ImageOps.exif_transpose(image)
            return image
        except Exception as e:
            st.error(f"Error fixing orientation: {e}")
            return None
    fixed_image = fix_orientation(uploaded_file)
        
    # image = Image.open(fixed_image)
    st.image(fixed_image, caption='Imagem carregada.', width=200)

    if model_url and model:
        st.write("Classificando...")
        def predict_image(image):
            # img = load_img(image, target_size=(150, 150))
            img = image.resize((150, 150))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0
            preds = model.predict(x)
            return preds
        preds = predict_image(fixed_image)
        # Read the JSON file
        with open(class_url, 'r') as file:
            class_indices = json.load(file)
        prediction_class = next(key for key, value in class_indices.items() if value == np.argmax(preds))
        # Read the JSON file
        with open(description_url, 'r') as file:
            descriptions_indices = json.load(file)
            
        resultados = descriptions_indices[prediction_class]
        if preds.max() > 0.3:
            st.write(f"Resultado: {resultados}")
            st.write(f"Confiança: {round(preds.max()*100,2)} %")
        else:
            st.write("Pouca confiança na previsão. Se for possível, tente usar fundo branco e apenas um teste por vez.")
    else:
        st.warning("Carregar uma imagem para realizar a predição.")
else:
    st.warning("Carregar uma imagem para realizar a predição.")
