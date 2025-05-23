from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load LSTM model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prices = [float(request.form[f"price{i}"]) for i in range(10)]
    prices_scaled = scaler.transform(np.array(prices).reshape(-1, 1))
    X_input = np.reshape(prices_scaled, (1, 10, 1))
    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    return render_template('index.html', prediction_text=f"Predicted Gold Price: ₹{predicted_price:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
