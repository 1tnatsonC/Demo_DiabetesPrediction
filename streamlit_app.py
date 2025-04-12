import streamlit as st
import pickle
import numpy as np

# Load the exported model
with open('xgbc_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler (if used for standardization)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Diabetes Prediction")

# User inputs
glucose = st.text_input("Enter Glucose Level")
insulin = st.text_input("Enter Insulin Level")
bmi = st.text_input("Enter BMI")
skin_thickness = st.text_input("Enter Skin Thickness")

if st.button("Predict"):
    try:
        # Validate user inputs
        inputs = [glucose, insulin, bmi, skin_thickness]
        if any(not input.isnumeric() and not input.replace('.', '', 1).isdigit() for input in inputs):
            raise ValueError("All inputs must be valid numbers (integer or float).")

        # Prepare input data
        user_input = np.array([float(glucose), float(insulin), float(bmi), float(skin_thickness)]).reshape(1, -1)
        std_data = scaler.transform(user_input)  # Standardize input
        prediction = model.predict(std_data)

        # Display result
        if prediction[0] == 0:
            st.success("The user does not have Diabetes.")
        else:
            st.warning("The user may have Diabetes.")
    except ValueError as ve:
        st.error(f"Input Error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
