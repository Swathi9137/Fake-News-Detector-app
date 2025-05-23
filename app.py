import streamlit as st
import joblib


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.title("📰 Fake News Detector")
st.write("Enter a news headline to check if it's REAL or FAKE.")

user_input = st.text_area("News Headline")

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter a news headline to check.")
    else:
        cleaned_input = user_input.lower()
        vec_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vec_input)[0]
        label = "FAKE" if prediction == 1 else "REAL"
        st.success(f"Prediction: {label}")



