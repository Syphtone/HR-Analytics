import streamlit as st
import joblib
import numpy as np

# Load the saved Decision Tree model
model = joblib.load('best_model.pkl')

# App layout
st.title("Absenteeism Prediction App")
st.markdown("Enter the input features to predict:")

# Input fields
Age = st.number_input("Age", value=0.0)
LengthService = st.number_input("LengthService", value=0.0)
Gender = st.number_input("Gender", value=0.0)

# Create input array
input_data = np.array([[Age, LengthService, Gender]])

# Submit button
if st.button("Submit"):
    # Make prediction
    prediction = model.predict(input_data)[0]
    # Display prediction
    st.subheader(f"Output: {prediction}")

# Flag button (placeholder)
if st.button("Flag"):
    st.warning("Flagging functionality is not implemented.")