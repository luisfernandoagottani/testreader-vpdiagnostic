import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requests
import h5py
from io import BytesIO
import pickle

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
        model_url = "https://github.com/luisfernandoagottani/testreader-vpdiagnostic/edit/master/pif/pif_20240607.pkl"  # Replace with the actual URL of your joblib file

@st.cache_resource
def load_model(url):
    # model_file = BytesIO(requests.get(url).content)
    # Load the model from the file-like object using h5py
    # model = pickle.load(model_file)
    # model = joblib.load(model_file)
    with open("./pif/pif_20240607.pkl", "rb") as pickle_in:
        model = pickle.load(pickle_in)
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
        preds = predict_image(image_path)
        
        st.write(f"Prediction: {np.argmax(preds)}")
    else:
        st.warning("Please upload a model file to make predictions.")
else:
    st.warning("Please upload an image to classify.")
