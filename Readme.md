# 🕵️ Fake Job Posting Detector (NLP + ML)

This project uses Natural Language Processing (NLP) and Machine Learning to detect fraudulent job postings.

## 📌 Problem
With the rise of fake job offers online, companies and job seekers need tools to filter real vs fake listings. This model predicts the legitimacy of job postings using text classification.

## 📂 Dataset
- Source: [Kaggle - Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

## 🧠 Techniques Used
- Data Cleaning & Preprocessing
- NLP (TF-IDF, Stopword Removal, Stemming)
- ML Models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Accuracy, Precision, Recall, ROC-AUC
- Streamlit App for deployment

## 💡 Result
Achieved high accuracy and AUC in detecting fake job listings using Random Forest.

## 🚀 Run Locally

```bash
git clone https://github.com/Harshit-0/Fake-Job-ML
cd Fake_Job_Pred
pip install -r requirements.txt
streamlit run app.py
```
