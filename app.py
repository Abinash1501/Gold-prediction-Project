import streamlit as st

st.title("Gold Price Prediction")

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Gold Price Prediction (LSTM)")

# User input
input_data = st.text_input("Enter 5 previous days' prices separated by commas", "48000,48100,47950,48050,48200")

if st.button("Predict"):
    try:
        values = np.array([float(x) for x in input_data.split(",")])
        if len(values) != 5:
            st.error("Please enter exactly 5 values.")
        else:
            scaled_input = scaler.transform(values.reshape(-1, 1)).reshape(1, 5, 1)
            prediction = model.predict(scaled_input)
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            st.success(f"Predicted next price: ₹{predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
