import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

# ----------------------------
# Load model and tokenizer
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fake_news_model.h5")
    return model

@st.cache_resource
def load_tokenizer():
    # If you have a saved tokenizer file (tokenizer.pkl or similar), load it
    with open("fake_news_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# ----------------------------
# Predict function
# ----------------------------
def predict_news(text, model, tokenizer, max_len=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)
    label = "🟢 Real News" if pred[0][0] > 0.5 else "🔴 Fake News"
    return label, float(pred[0][0])

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection using LSTM")
st.write("This app uses an LSTM-based neural network to detect whether a given news statement is **fake** or **real**.")

model = load_model()
tokenizer = load_tokenizer()

# User input
text_input = st.text_area("✍️ Enter news text here:", height=150)

if st.button("🔍 Detect"):
    if text_input.strip():
        label, score = predict_news(text_input, model, tokenizer)
        st.subheader("Result:")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter some text to analyze.")

# Optional: Upload text file
uploaded_file = st.file_uploader("Or upload a .txt file to analyze multiple lines", type=["txt"])
if uploaded_file is not None:
    text_data = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded File Content", text_data, height=200)
    if st.button("Analyze Uploaded File"):
        lines = text_data.strip().split("\n")
        results = []
        for line in lines:
            if line.strip():
                label, score = predict_news(line, model, tokenizer)
                results.append((line, label, score))
        st.write("### Results:")
        for line, label, score in results:
            st.write(f"**Text:** {line}")
            st.write(f"→ {label} (Confidence: {score:.2f})")
            st.markdown("---")

# Download section (optional)
if 'results' in locals() and len(results) > 0:
    import pandas as pd
    df = pd.DataFrame(results, columns=["Text", "Prediction", "Confidence"])
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download Results as CSV", csv, "fake_news_results.csv", "text/csv")
