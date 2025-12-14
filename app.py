import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="COVID X-ray Classifier", page_icon="🩻", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_mobilenetv2_model.keras")

model = load_model()

IMG_SIZE = (224, 224)
CLASS_NAMES = ["COVID", "Normal"]  # عدّلها إذا ترتيبك مختلف أثناء التدريب

st.title("🩻 COVID X-ray Classification (MobileNetV2)")
st.write("Upload an X-ray image and the model will predict the class.")

threshold = st.slider("Decision threshold (label=1)", 0.1, 0.9, 0.5, 0.05)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # ✅ لا تقسّم /255 لأن داخل النموذج Rescaling

    prob_label1 = float(model.predict(x, verbose=0)[0][0])  # P(CLASS_NAMES[1])
    pred_label = 1 if prob_label1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_label]
    confidence = (prob_label1 if pred_label == 1 else (1 - prob_label1)) * 100

    st.subheader("Result")
    st.write(f"**Raw sigmoid P(label=1)** = `{prob_label1:.4f}`")
    st.write(f"**Prediction:** `{pred_name}`")
    st.write(f"**Confidence:** `{confidence:.2f}%`")
