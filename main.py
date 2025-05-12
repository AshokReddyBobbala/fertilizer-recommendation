import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras as k

# Load the model
model = k.models.load_model("model_new-1.h5")

# Fertilizer recommendation function

def recommend_fertilizer(class_idx):
    recommendations  = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
}

    return recommendations.get(class_idx, "Unknown class")

# Image preprocessing function
def process_image(uploaded_file):
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

# Streamlit UI
st.title("🌱 Fertilizer Recommendation System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])


if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)
    image = process_image(uploaded_file)
    pred = model.predict(np.expand_dims(image, axis=0))

    class_idx = np.argmax(pred)
    st.write("### 🧪 Fertilizer Recommendation:")
    st.success(recommend_fertilizer(class_idx))
