import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re

# --- Configuration ---
# Ensure these filenames match the files uploaded to your cloud repository
MODEL_PATH = 'fake_news_detection.h5'
TOKENIZER_PATH = 'fake_news_tokenizer.pkl'
MAX_SEQUENCE_LENGTH = 250 # Based on analysis of the training notebook

# Set page configuration early
st.set_page_config(page_title="LSTM Fake News Detector", layout="wide")

@st.cache_resource
def load_assets():
    """Loads the model and tokenizer only once, using correct filenames."""
    try:
        # Load the LSTM Model
        st.write(f"Loading model from: {MODEL_PATH}")
        # The .h5 file extension implies loading a Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load the Tokenizer
        st.write(f"Loading tokenizer from: {TOKENIZER_PATH}")
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure '{e.filename}' is in your deployment directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        st.stop()

def clean_text(text):
    """
    Applies common pre-processing steps: lowercasing and removing non-alphanumeric characters.
    This should align with the pre-processing used during training.
    """
    text = text.lower()
    # Remove punctuation, numbers, and multiple spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_fake_news(text, model, tokenizer):
    """
    Tokenizes, pads, and predicts the class of the input text.
    """
    # 1. Clean the input text
    cleaned_text = clean_text(text)
    
    # 2. Convert text to sequence of integers
    # texts_to_sequences expects a list, even for a single text
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Pad the sequence to the maximum length
    padded_sequence = pad_sequences(
        sequence, 
        maxlen=MAX_SEQUENCE_LENGTH, 
        padding='post', 
        truncating='post'
    )
    
    # 4. Get prediction from the model
    # Model expects a batch of inputs (even if it's a batch of 1)
    prediction = model.predict(padded_sequence, verbose=0)
    
    # The output is a single probability for the positive class (1, which is FAKE)
    probability_fake = prediction[0][0] 
    
    # Determine the class based on a 0.5 threshold
    if probability_fake >= 0.5:
        return 'FAKE', probability_fake
    else:
        # The probability of being REAL (0) is 1 - P(FAKE)
        return 'REAL', 1 - probability_fake 

# --- Streamlit App UI ---

st.title("📰 LSTM-Powered Fake News Detector")
st.markdown(
    """
    This application uses a deep learning model (LSTM) to classify text as either **REAL** or **FAKE** news. 
    To deploy this to the cloud, ensure your `app.py`, `requirements.txt`, `fake_news_detection.h5`, 
    and `fake_news_tokenizer.pkl` files are all in the same repository.
    """
)

# Load the model and tokenizer
# This function uses st.cache_resource, so it runs only on initial startup
model, tokenizer = load_assets()

# Text Area for user input
user_input = st.text_area(
    "Paste the News Article or Title Here",
    height=300,
    placeholder="Example: 'NASA confirms that the moon is made entirely of green cheese, shocking the scientific community.'"
)

# Prediction button
if st.button("🔮 Predict News Credibility", type="primary"):
    if not user_input or len(user_input.strip()) < 5:
        st.warning("Please enter a longer piece of text to analyze.")
    else:
        with st.spinner('Analyzing text and predicting...'):
            label, confidence = predict_fake_news(user_input, model, tokenizer)
        
        st.subheader("Prediction Result:")
        
        # Display the result with appropriate styling
        if label == 'FAKE':
            st.error(f"🚨 **CLASSIFIED AS FAKE NEWS**")
            st.markdown(f"The model is **{confidence:.1%}** confident that this text is **FAKE**.")
            st.snow()
        else: # REAL
            st.success(f"✅ **CLASSIFIED AS REAL NEWS**")
            st.markdown(f"The model is **{confidence:.1%}** confident that this text is **REAL**.")
        
        # Display raw confidence score
        st.markdown("---")
        st.caption(f"Raw Model Confidence (P(FAKE)): {confidence if label=='FAKE' else 1-confidence:.4f}")

st.sidebar.header("Model Details")
st.sidebar.info(
    "This is a sequential deep learning model built with a LSTM layer, trained on news datasets."
)
st.sidebar.markdown(f"**Tokenizer Vocabulary Size:** `{len(tokenizer.word_index) + 1}`")
st.sidebar.markdown(f"**Max Input Length (Padding):** `{MAX_SEQUENCE_LENGTH}` tokens")
st.sidebar.markdown(f"**TensorFlow Version:** `{tf.__version__}`")
