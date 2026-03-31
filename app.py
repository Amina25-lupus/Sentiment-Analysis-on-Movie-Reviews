

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data for deployment
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- 1. Load Assets ---
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    lr = pickle.load(open('lr_model.pkl', 'rb'))
    nb = pickle.load(open('nb_model.pkl', 'rb'))
    return tfidf, lr, nb

tfidf, lr_model, nb_model = load_models()

# --- 2. Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# --- 3. UI Setup ---
st.set_page_config(page_title="Sentiment Model Comparison", layout="wide")
st.title("🎬 Movie Review Sentiment: Model Comparison")
st.write("Enter a review below to see how **Logistic Regression** and **Naive Bayes** compare.")

# Input area
user_input = st.text_area("Write your movie review here:", height=150)

if st.button("Compare Models"):
    if user_input.strip():
        # Process Input
        cleaned = preprocess_text(user_input)
        vectorized = tfidf.transform([cleaned])
        
        # Predictions
        lr_pred = lr_model.predict(vectorized)[0]
        nb_pred = nb_model.predict(vectorized)[0]
        
        lr_label = "Positive ✅" if lr_pred == 1 else "Negative ❌"
        nb_label = "Positive ✅" if nb_pred == 1 else "Negative ❌"

        # Display Side-by-Side
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Logistic Regression")
            st.metric("Sentiment", lr_label)
            st.info("Accuracy: 89%")
            
        with col2:
            st.header("Naive Bayes")
            st.metric("Sentiment", nb_label)
            st.info("Accuracy: 85%")

        # Feedback if they disagree
        if lr_pred != nb_pred:
            st.warning("⚠️ The models disagree on this review! This often happens with sarcasm or complex language.")
        else:
            st.success("🤝 Both models agree on this sentiment.")
            
    else:
        st.error("Please enter a review to analyze.")

# --- 4. Sidebar Comparison Stats ---
st.sidebar.title("Model Performance Report")
st.sidebar.write("**Logistic Regression**")
st.sidebar.progress(89)
st.sidebar.write("**Naive Bayes**")
st.sidebar.progress(85)
st.sidebar.markdown("""
---
### Why the difference?
Logistic Regression looks at the **weight** of words, while Naive Bayes looks at the **probability** of words appearing. Logistic Regression is generally better at handling the context of movie reviews.
""")
