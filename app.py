import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and the scaler
model = joblib.load('logistic_regression_model.joblib')

# Assuming a scaler was used during training and saved as 'scaler.joblib'
# If no scaler was used, you can remove this line and the scaling step below.
try:
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    print("Warning: 'scaler.joblib' not found. Ensure the model was trained without scaling or the scaler file is present.")
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert input data to pandas DataFrame
        # Ensure the column order matches the training data
        # The diabetes dataset has 8 features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        # Example expected input: {"Pregnancies": 2, "Glucose": 100, "BloodPressure": 70, "SkinThickness": 30, "Insulin": 50, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30}

        input_df = pd.DataFrame([data])

        # Apply the scaler if it was loaded
        if scaler:
            input_scaled = scaler.transform(input_df)
            # Convert back to DataFrame to maintain feature names if needed for debugging, or just use the array
            input_df_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
        else:
            input_df_scaled = input_df # Use original if no scaler

        # Make prediction using the (potentially scaled) input
        prediction = model.predict(input_df_scaled)
        prediction_proba = model.predict_proba(input_df_scaled)

        # Return the prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_probability_class_0': float(prediction_proba[0][0]),
            'prediction_probability_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == "main":
  app.run(debug=True)

# To run the app, you would typically use `app.run()`
# However, in Colab, this will block the execution.
# For testing within Colab, you can use ngrok or similar to expose the server.
# For a simple local test in a separate Python script, you would run:
# if __name__ == '__main__':
#    app.run(debug=True)
