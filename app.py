import os
import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Load model
model = joblib.load("logistic_regression_model.joblib")

# Load scaler
try:
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    print("Warning: scaler.joblib not found.")
    scaler = None

# Create Flask app
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "Diabetes Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        feature_columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        input_df = input_df[feature_columns]

        # Scale input
        if scaler is not None:
            input_data = scaler.transform(input_df)
        else:
            input_data = input_df

        # Prediction
        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_class_0": float(probability[0][0]),
            "probability_class_1": float(probability[0][1])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# Start Flask server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
