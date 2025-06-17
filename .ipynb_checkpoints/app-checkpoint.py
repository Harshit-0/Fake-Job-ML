import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model and vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🕵️ Fake Job Posting Detector")
st.write("Paste job description or full posting below:")

input_text = st.text_area("Job Posting Text", height=250)

if st.button("Check if Fake"):
    if input_text.strip() == "":
        st.warning("Please paste some job content.")
    else:
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        label = "🚫 Fake Job Posting" if prediction == 1 else "✅ Real Job Posting"
        st.subheader(f"Result: {label}")
