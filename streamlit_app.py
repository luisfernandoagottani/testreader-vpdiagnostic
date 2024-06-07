import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page title
st.set_page_config(page_title='Image Diagnosis Prediction', page_icon='🩺')
st.title('🩺 Image Diagnosis Prediction')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to upload an image and get a diagnosis prediction using a pre-trained model.')

# Sidebar for model selection
with st.sidebar:
    st.header('Model')
    model_file = st.file_uploader("Upload a Keras model (.h5 file)", type=["h5"])
    if model_file is not None:
        model = tf.keras.models.load_model(model_file)

# Main section for image upload and prediction
st.header('Upload an Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if model_file is not None:
        st.write("Classifying...")
        # Preprocess the image to match the model's expected input
        img_array = np.array(image.resize((224, 224)))  # Resize to model's input size
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        st.write(f"Prediction: {predicted_class}")
    else:
        st.warning("Please upload a Keras model file (.h5) to make predictions.")
else:
    st.warning("Please upload an image to classify.")
