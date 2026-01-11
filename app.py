import streamlit as st
import numpy as np
import joblib

# Load saved objects
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient medical details to predict diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
skin = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=100, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([
        pregnancies, glucose, bp, skin,
        insulin, bmi, dpf, age
    ]).reshape(1, -1)

    # Preprocessing
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)[0]

    if prediction == 1:
        st.error("❌ Patient is Diabetic")
    else:
        st.success("✅ Patient is Non-Diabetic")
