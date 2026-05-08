# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib  # For loading the scikit-learn model
import os

app = Flask(__name__)

# Load the trained model and scaler
model = None
scaler = None
feature_names = [
    'sleep_quality',
    'exercise_minutes',
    'hours_worked',
    'social_interactions',
    'work_load_score',
    'heart_rate'
]

try:
    # Try to load model
    if os.path.exists('models/stress_model.pkl'):
        model = joblib.load('models/stress_model.pkl')
        print("✓ Model loaded successfully")
    else:
        print("⚠ Model file not found. Please train your model first.")
    
    # Try to load scaler if it exists (from advanced training)
    if os.path.exists('models/stress_scaler.pkl'):
        scaler = joblib.load('models/stress_scaler.pkl')
        print("✓ Scaler loaded successfully (using advanced model)")
    else:
        print("⚠ Scaler not found (using basic model)")
        
except Exception as e:
    print(f"✗ Error loading model/scaler: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please train your model first.'}), 503

    try:
        data = request.get_json()
        
        # Extract features from the JSON data and ensure correct order
        input_features = [float(data.get(feature, 0)) for feature in feature_names]

        # Check if all required features are present
        if None in input_features or any(v == 0 and data.get(feature_names[i]) is None for i, v in enumerate(input_features)):
            missing_features = [feature_names[i] for i, val in enumerate(input_features) if data.get(feature_names[i]) is None]
            if missing_features:
                return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400

        # Convert to DataFrame suitable for the model
        input_df = pd.DataFrame([input_features], columns=feature_names)
        
        # Apply scaler if available (advanced model)
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            input_df_model = pd.DataFrame(input_scaled, columns=feature_names)
        else:
            input_df_model = input_df

        # Make prediction
        prediction = model.predict(input_df_model)
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(input_df_model)
            confidence = float(max(prediction_proba[0])) * 100
        except:
            confidence = 0

        prediction_value = int(prediction[0])

        # Map prediction to user-friendly stress level
        stress_levels = {
            0: "Low Stress",
            1: "Moderate Stress", 
            2: "High Stress",
            3: "Very High Stress"
        }
        
        stress_level = stress_levels.get(prediction_value, f"Stress Level: {prediction_value}")

        return jsonify({
            'prediction': stress_level,
            'confidence': confidence,
            'raw_value': prediction_value
        })

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    # Debug=True is useful for development, but set to False for production
    app.run(debug=True)