import streamlit as st
from model.classifier import SentimentClassifier
from utils.sample_data import load_sample_data

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = SentimentClassifier()

# App header
st.title("📊 Sentiment Analysis App")
st.markdown("This app analyzes customer reviews and classifies them as positive or negative.")


# Sidebar
with st.sidebar:
    st.header("Model Training")
    if st.button("Load Sample Data & Train Model"):
        with st.spinner("Training model..."):
            df = load_sample_data()
            st.session_state.classifier.train(df['text'], df['sentiment'])
            st.success("Model trained successfully!")

# Main content
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
            prediction = st.session_state.classifier.predict(review_text)
            probabilities = st.session_state.classifier.get_prediction_proba(review_text)

            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probabilities[1] if prediction == 1 else probabilities[0]

            st.success(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")

# Footer (retained from original)
st.markdown("""
---
<div style="text-align: center; padding: 16px;">
    <p>Built with Streamlit • Scikit-learn • NLTK</p>
</div>
""", unsafe_allow_html=True)