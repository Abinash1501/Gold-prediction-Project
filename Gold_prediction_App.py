import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the model
model = joblib.load('xgboost_gold_model.joblib')

# App title
st.title("Gold Price Prediction with XGBoost")

# Input section
st.header("Enter input for next day prediction")

# Example: 1-day lag input
price_today = st.number_input("Enter the latest gold price", min_value=0.0)

if st.button("Predict"):
    X_input = np.array([[price_today]])  # Adjust shape if needed
    prediction = model.predict(X_input)[0]
    st.success(f"Predicted Gold Price for Next Day: ₹ {prediction:.2f}")
