import pickle
import numpy as np
import streamlit as st


model_path = 'model.pkl'
scalar_path = 'scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(scalar_path, 'rb') as scalar_file:
    scalar = pickle.load(scalar_file)

st.title("Diabetes Prediction App")


Pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, step=1)
BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1)
SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
Insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, step=1)
BMI = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function (score)", min_value=0.0, max_value=2.5, step=0.001, format="%.3f")
Age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)

if st.button('Predict'):
    
    data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
    data = scalar.transform(data)
    
    
    prediction = model.predict(data)
    
    
    if prediction[0] == 1:
        st.error("The model predicts that this person **is Diabetic**.")
    else:
        st.success("The model predicts that this person **is not Diabetic**.")
