import streamlit as st
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load('models/news_classifier_pipeline.pkl')

# Streamlit app UI
st.set_page_config(page_title="News Classifier", layout="centered")
st.title("📰 News Category Classifier")
st.markdown("Enter a news article content below to predict its category.")

# Input form
with st.form("news_form"):
    content = st.text_area("News Content", height=200)
    submitted = st.form_submit_button("Classify")

# Prediction logic
if submitted:
    if content.strip() == "":
        st.warning("Please enter some content.")
    else:
        prediction = pipeline.predict([content])[0]
        st.success(f"Predicted Category: **{prediction}**")

