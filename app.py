import streamlit as st
import joblib
import numpy as np

# Load the pre-trained SVM model and scaler
svm_model = joblib.load('svm_model.pkl')   # Load your SVM model
scaler = joblib.load('scaler.pkl')         # Load your scaler

# Frontend: User input form
st.title("SVM Model Prediction")

# User inputs for the features (now with 3 features)
feature_1 = st.number_input("Enter Feature 1 (e.g., Gender)", min_value=0.0, max_value=100.0, step=0.1)
feature_2 = st.number_input("Enter Feature 2 (e.g., Age)", min_value=0.0, max_value=100.0, step=0.1)
feature_3 = st.number_input("Enter Feature 3 (e.g., Estimated Salary)", min_value=0.0, max_value=200000.0, step=0.1)  # Third feature input

# Function to make predictions
def make_prediction(features):
    # Preprocess the features using the loaded scaler
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    
    # Predict using the SVM model
    prediction = svm_model.predict(scaled_features)
    
    return prediction

# Prediction when the button is pressed
if st.button("Predict"):
    prediction = make_prediction([feature_1, feature_2, feature_3])  # Include feature_3
    if prediction == 1:
        st.write("Prediction: Purchased")  # Interpreting the model's prediction
    else:
        st.write("Prediction: Not Purchased")
