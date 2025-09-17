from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"]
)

# Load the model
try:
    model = load_model("tesla_stock_model.h5")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Load CSV data and prepare the scaler
try:
    data = pd.read_csv("tesla_stock_data.csv", usecols=["Open", "High", "Low", "Volume", "Close"])
    data_X = data[["Open", "High", "Low", "Volume"]].values
    data_y = data["Close"].values

    X_train, _, y_train, _ = train_test_split(data_X, data_y, test_size=0.9, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)

except Exception as e:
    print("Error loading CSV or fitting scaler:", e)
    scaler = None
    y_train_mean = 0
    y_train_std = 1

@app.route("/")
def index():
    return "Tesla Stock Predictor is running on Render 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        print("Received:", input_data)

        features = [[
            float(input_data["open"]),
            float(input_data["high"]),
            float(input_data["low"]),
            float(input_data["volume"])
        ]]

        if scaler is None or model is None:
            return jsonify({"error": "Model or scaler not loaded"}), 500

        features_scaled = scaler.transform(features)
        pred_norm = model.predict(features_scaled).flatten()[0]
        predicted_close = pred_norm * y_train_std + y_train_mean

        return jsonify({"predicted": round(predicted_close, 2)})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)