import streamlit as st
import joblib
import numpy as np
import os

# Absolute path fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
bp = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    data = np.array([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1,-1)
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    prediction = model.predict(data_pca)[0]
    if prediction==1:
        st.error("❌ Patient is Diabetic")
    else:
        st.success("✅ Patient is Non-Diabetic")
