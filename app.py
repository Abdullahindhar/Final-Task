# app.py

import streamlit as st
import numpy as np
import joblib

# Load saved objects
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("best_model.pkl")

st.title("Diabetes Prediction App")
st.write("Enter patient medical details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Prepare input
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Scaling
    input_scaled = scaler.transform(input_data)
    
    # PCA transformation
    input_pca = pca.transform(input_scaled)
    
    # Prediction
    prediction = model.predict(input_pca)
    
    # Output
    if prediction[0] == 1:
        st.error("The patient is Diabetic.")
    else:
        st.success("The patient is Non-Diabetic.")
