import streamlit as st
import pandas as pd
from model.preprocessor import TextPreprocessor
from model.classifier import SentimentClassifier
from utils.sample_data import load_sample_data
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)

# Load custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = SentimentClassifier()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()

# App header
st.title("📊 Sentiment Analysis App")
st.markdown("""
This app analyzes customer reviews and classifies them as positive or negative using 
machine learning techniques.
""")

# Sidebar
with st.sidebar:
    st.header("Model Training")
    if st.button("Load Sample Data & Train Model"):
        with st.spinner("Loading sample data..."):
            df = load_sample_data()
            
        # Preprocess the data
        with st.spinner("Preprocessing text..."):
            X = df['text'].apply(st.session_state.preprocessor.preprocess_text)
            y = df['sentiment']
            
        # Train the model
        with st.spinner("Training model..."):
            st.session_state.classifier.train(X, y)
            st.success("Model trained successfully!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Review Analysis")
    review_text = st.text_area(
        "Enter your review text:",
        height=150,
        placeholder="Type or paste your review here..."
    )

    if st.button("Analyze Sentiment"):
        if not review_text:
            st.warning("Please enter some text to analyze.")
        elif not st.session_state.classifier.is_trained:
            st.warning("Please train the model first using the sidebar.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Preprocess the input text
                processed_text = st.session_state.preprocessor.preprocess_text(review_text)
                
                # Get prediction and probability
                prediction = st.session_state.classifier.predict(processed_text)
                probabilities = st.session_state.classifier.get_prediction_proba(processed_text)
                
                # Display results
                sentiment = "Positive" if prediction == 1 else "Negative"
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
                
                st.markdown(f"""
                <div class="metrics-container">
                    <h3>Analysis Results</h3>
                    <p><strong>Sentiment:</strong> {sentiment}</p>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.subheader("Model Information")
    if st.session_state.classifier.is_trained:
        st.markdown("""
        <div class="metrics-container">
            <h4>Model Status</h4>
            <p>✅ Model is trained and ready to use</p>
            <h4>Features</h4>
            <ul>
                <li>TF-IDF Vectorization</li>
                <li>Logistic Regression Classifier</li>
                <li>NLTK Preprocessing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Model needs to be trained. Use the sidebar to train the model.")

# Footer
st.markdown("""
---
<div style="text-align: center; padding: 16px;">
    <p>Built with Streamlit • Scikit-learn • NLTK</p>
</div>
""", unsafe_allow_html=True)
