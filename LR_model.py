import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("lungcap_linear_regression.pkl", "rb"))

st.set_page_config(page_title="Lung Capacity Predictor", layout="centered")

st.title("🫁 Lung Capacity Prediction App")
st.write("Enter patient details to predict lung capacity")

# Inputs
age = st.number_input("Age", min_value=3, max_value=25, value=10)
height = st.number_input("Height (inches)", min_value=40.0, max_value=85.0, value=60.0)

smoke = st.selectbox("Smoker", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female"])
caesarean = st.selectbox("Caesarean Birth", ["No", "Yes"])

# Encoding
smoke_val = 1 if smoke == "Yes" else 0
gender_val = 1 if gender == "Male" else 0
caesarean_val = 1 if caesarean == "Yes" else 0

# Prediction
if st.button("Predict Lung Capacity"):
    input_data = np.array([[age, height, smoke_val, gender_val, caesarean_val]])
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Lung Capacity: {prediction[0]:.2f}")
    st.info("Model Accuracy: 88%")

col1, col2 = st.columns(2)