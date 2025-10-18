import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# ----------------------------
# Streamlit Page Settings
# ----------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection using LSTM")
st.write("This app uses a pre-trained LSTM model to detect whether a given news text is **Fake** or **Real**.")

# ----------------------------
# Helper Functions
# ----------------------------
@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def load_tokenizer(path):
    try:
        with open(path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        return None

def predict_news(text, model, tokenizer, max_len=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)
    label = "🟢 Real News" if pred[0][0] > 0.5 else "🔴 Fake News"
    return label, float(pred[0][0])

# ----------------------------
# Model and Tokenizer Loading
# ----------------------------
model_path = "fake_news_model.h5"
tokenizer_path = "fake_news_tokenizer.pkl"

# Check if model and tokenizer exist
if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    st.warning("⚠️ Model or tokenizer file not found. Please upload them below:")

    uploaded_model = st.file_uploader("Upload Model (.h5)", type=["h5"])
    uploaded_tokenizer = st.file_uploader("Upload Tokenizer (.pkl)", type=["pkl"])

    if uploaded_model is not None and uploaded_tokenizer is not None:
        with open("fake_news_model_uploaded.h5", "wb") as f:
            f.write(uploaded_model.getbuffer())
        with open("fake_news_tokenizer_uploaded.pkl", "wb") as f:
            f.write(uploaded_tokenizer.getbuffer())

        st.success("✅ Files uploaded successfully! You can now use the model.")
        model = load_model("fake_news_model_uploaded.h5")
        tokenizer = load_tokenizer("fake_news_tokenizer_uploaded.pkl")
    else:
        st.stop()
else:
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

# ----------------------------
# Prediction UI
# ----------------------------
if model and tokenizer:
    st.subheader("🔍 Enter News Text Below:")
    text_input = st.text_area("Type or paste the news article here:", height=150)

    if st.button("Detect Fake News"):
        if text_input.strip():
            label, confidence = predict_news(text_input, model, tokenizer)
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence:.2f}")
        else:
            st.warning("Please enter some text to analyze.")
