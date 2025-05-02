import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    try:
        nltk.download('stopwords')
        return stopwords.words('english')
    except Exception as e:
        st.error(f"Error downloading stopwords: {e}")
        return []

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = vectorizer.transform([text])
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Create a styled card for tweet sentiment
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    return f"""
    <div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <h5 style='color: white;'>{sentiment} Sentiment</h5>
        <p style='color: white;'>{tweet_text}</p>
    </div>
    """

# Main app logic
def main():
    st.title("Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    if not model or not vectorizer:
        st.error("Error loading necessary components.")
        return

    option = st.radio("Choose analysis type", ["Input text"])
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()