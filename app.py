import nltk
nltk.download('stopwords')

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# 1. Setup Page Config
st.set_page_config(page_title="AI Movie Critic", page_icon="🎬")

# 2. Load the Model and Vectorizer
@st.cache_resource
def load_assets():
    with open('movie_sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_assets()

# 3. Preprocessing Function (Must match your training code!)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# 4. User Interface
st.title("🎬 AI Movie Sentiment Analyzer")
st.write("Enter a movie review below, and the AI will tell you if it's Positive or Negative.")

user_input = st.text_area("Write your review here...", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Step A: Preprocess
        cleaned_text = preprocess_text(user_input)
        
        # Step B: Vectorize (Transform only, do NOT fit)
        vectorized_input = tfidf.transform([cleaned_text])
        
        # Step C: Predict
        prediction = model.predict(vectorized_input)[0]
        probability = model.predict_proba(vectorized_input)[0]
        
        # Step D: Display Results
        if prediction == 1:
            st.success(f"**Result: Positive Sentiment** ✅ (Confidence: {probability[1]:.2%})")
            st.balloons()
        else:
            st.error(f"**Result: Negative Sentiment** ❌ (Confidence: {probability[0]:.2%})")
            
    else:
        st.warning("Please enter some text first!")

# 5. Sidebar Info
st.sidebar.title("About the Model")
st.sidebar.info("This app uses a Logistic Regression model trained on 50,000 IMDB reviews. It achieves 89% accuracy.")