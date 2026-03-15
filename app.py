import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Fruit Recognition",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 AI Fruit Recognition & Calorie Estimation")
st.markdown("Upload a fruit image and the AI will predict the fruit and its calories per 100g.")

# -----------------------------
# Model Download
# -----------------------------
MODEL_PATH = "fruit_calorie_ai_model_finetuned.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... ⏳"):
        url = "https://drive.google.com/uc?id=1Nmg485B9wnRAV4Cyp4u66HyxuenvtBza"
        gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# Load JSON Files
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("final_calorie_map.json", "r") as f:
    calorie_map = json.load(f)

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Fruit Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image with AI... 🤖"):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    predicted_class = class_names[predicted_index]

    calories = calorie_map.get(predicted_class, "Not Available")

    st.success(f"🍏 Predicted Fruit: **{predicted_class}**")
    st.info(f"🔥 Calories (per 100g): **{calories} kcal**")
    st.write(f"🎯 Prediction Confidence: **{confidence:.2f}%**")

    st.progress(int(confidence))

else:
    st.info("Please upload a fruit image to start prediction.")
