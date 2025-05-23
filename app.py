import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Gold Price Prediction (LSTM Model)")
st.write("Enter the last 10 days of gold prices to predict the next day's price:")

# Collect input prices
prices = []
for i in range(10):
    price = st.number_input(f"Day {i+1} Price", min_value=0.0, step=0.01, format="%.2f", key=f"price_{i}")
    prices.append(price)

if st.button("Predict"):
    if all(p > 0 for p in prices):
        prices_scaled = scaler.transform(np.array(prices).reshape(-1, 1))
        X_input = np.reshape(prices_scaled, (1, 10, 1))
        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        st.success(f"Predicted Gold Price: ₹{predicted_price:.2f}")
    else:
        st.error("Please enter all 10 prices (non-zero).")
