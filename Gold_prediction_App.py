import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
#import joblib
from datetime import datetime, timedelta



# Load model and scaler
model = load_model('lstm_gold_model.h5')
scaler = joblib.load('scaler.save')

# Load original data for sequence base
df = pd.read_csv('gold_data.csv', parse_dates=['date'])

sequence_length = 60

def create_input_sequence(data, seq_length):
    scaled = scaler.transform(data[-seq_length:].reshape(-1,1))
    return scaled.reshape(1, seq_length, 1)

st.title("Gold Price Forecasting with Tuned LSTM")

st.write("This app predicts gold prices for the next 365 days based on historical data.")

if st.button("Predict Next 365 Days"):
    data_prices = df['price'].values
    input_seq = create_input_sequence(data_prices, sequence_length)
    
    predictions_scaled = []
    input_for_pred = input_seq.copy()
    
    for _ in range(365):
        pred = model.predict(input_for_pred)[0][0]
        predictions_scaled.append(pred)
        input_for_pred = np.append(input_for_pred[:,1:,:], [[[pred]]], axis=1)
    
    preds = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1,1)).flatten()
    
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]
    
    result = pd.DataFrame({'Date': future_dates, 'Predicted Price': preds})
    st.write(result)
    
    st.line_chart(result.set_index('Date')['Predicted Price'])

