import streamlit as st
import pickle
import numpy as np
import pandas as pd


model = pickle.load(open('xgboost_gold_model.pkl', 'rb')

def forecast_xgboost(model, last_prices, n_steps=30, lag=5):
    lag_data = list(last_prices[-lag:])
    preds = []
    for _ in range(n_steps):
        features = np.array(lag_data[-lag:]).reshape(1, -1)
        next_val = model.predict(features)[0]
        preds.append(next_val)
        lag_data.append(next_val)
    return preds

st.title("Gold Price Forecasting with XGBoost")

st.write("Enter the most recent 5 gold prices (comma separated) to forecast the next 30 days.")

input_prices = st.text_input(
    "Last 5 Gold Prices (comma separated):",
    "1800,1805,1795,1810,1802"
)

if st.button("Forecast"):
    try:
        last_prices = [float(x.strip()) for x in input_prices.split(",")]
        if len(last_prices) < 5:
            st.error("Please enter at least 5 prices.")
        else:
            forecast = forecast_xgboost(model, last_prices, n_steps=30, lag=5)
            forecast_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=30)
            forecast_series = pd.Series(forecast, index=forecast_dates)

            st.line_chart(forecast_series)
            st.write("### Forecasted Gold Prices")
            st.dataframe(forecast_series.rename("Predicted Price"))

    except Exception as e:
        st.error(f"Error: {e}")
