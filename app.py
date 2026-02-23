import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


model = load_model('skin_cancer_cnn.h5')


def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img

# UI App
st.title("Skin Cancer Detection System")
st.markdown("""
This is a skin cancer detection application. Upload an image of a skin lesion to get started and model will give results""")

uploaded_image = st.file_uploader("choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    class_label, img = predict_skin_cancer(uploaded_image, model)

    st.image(uploaded_image, caption='Uploaded Image', width=500)
    st.write(f"Prediction: **{class_label}**")
