import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# -----------------------------
# Auto Download Model from Drive
# -----------------------------
MODEL_PATH = "fruit_calorie_ai_model_finetuned.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1Nmg485B9wnRAV4Cyp4u66HyxuenvtBza"
    st.write("Downloading model... Please wait ⏳")
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Load JSON Files
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("final_calorie_map.json", "r") as f:
    calorie_map = json.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍎 AI-Based Fruit Recognition & Calorie Estimation")
st.write("Upload a fruit image and get predicted fruit name with calories per 100g.")

uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    calories = calorie_map.get(predicted_class, "Not Available")

    st.success(f"Predicted Fruit: {predicted_class}")
    st.info(f"Calories (per 100g): {calories} kcal")
