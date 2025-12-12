import json, numpy as np, streamlit as st, tensorflow as tf
from PIL import Image

IMG = 224

st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="centered")
st.title("🗑️ Garbage Classification (6 Classes)")

@st.cache_resource
def load_assets():
    labels = json.load(open("labels.json"))
    model = tf.keras.models.load_model("garbage_tf.keras")
    return model, labels

def preprocess(img):
    img = img.convert("RGB").resize((IMG, IMG))
    x = np.array(img, dtype=np.float32)[None, ...]
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x

with st.spinner("Loading model... (first time may take ~20-60s)"):
    model, labels = load_assets()

file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        p = model.predict(preprocess(img), verbose=0)[0]
        top = np.argsort(-p)[:3]

    st.success(f"Prediction: **{labels[int(top[0])] }** (conf: **{float(p[top[0]]):.3f}**)")

    st.write("Top-3:")
    for i in top:
        st.write(f"- {labels[int(i)]}: {float(p[int(i)]):.3f}")
