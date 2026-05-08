# app.py

import os
import joblib
import pandas as pd
import numpy as np # Often imported for data manipulation, even if not explicitly used
from flask import Flask, request, jsonify, render_template

# --- Configuration ---
MODEL_DIR = 'models'
MODEL_FILE = 'stress_model.pkl'
# Ensure these feature names EXACTLY match what was used to train the model
FEATURE_NAMES = [
    'sleep_quality',
    'exercise_minutes',
    'hours_worked',
    'social_interactions',
    'work_load_score',
    'heart_rate'
]
# --- Flask Application Setup ---
app = Flask(__name__)

# --- Helper function to load model and features ---
def load_model_and_features():
    """Loads the trained model and defines expected feature names."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_model.py first.")
    
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model, FEATURE_NAMES

# Load model and features when the app starts
try:
    model, feature_names_for_prediction = load_model_and_features()
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    # In a real app, you might want to handle this more gracefully (e.g., return an error page)
    model = None # Ensure model is None if loading failed
    feature_names_for_prediction = []
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    model = None
    feature_names_for_prediction = []

# --- Routes ---
@app.route('/')
def index():
    """Renders the main prediction form."""
    return render_template('index.html', feature_names_for_display=FEATURE_NAMES) # Pass feature names for potential dynamic labels

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        data_received = request.get_json()
        if not data_received:
            return jsonify({'error': 'Invalid JSON received.'}), 400

        # Create a list of feature values in the correct order
        # Ensure the keys in data_received match your HTML form's 'name' attributes
        # and the order matches feature_names_for_prediction
        input_data_list = []
        for feature in feature_names_for_prediction:
            if feature in data_received:
                # Convert to float, handling potential errors
                try:
                    value = float(data_received[feature])
                    input_data_list.append(value)
                except (ValueError, TypeError):
                    return jsonify({'error': f"Invalid input for {feature}. Please provide a number."}), 400
            else:
                # This should ideally not happen if HTML form names match FEATURE_NAMES
                return jsonify({'error': f"Missing feature in input data: {feature}"}), 400
        
        # Convert the list to a numpy array (or pandas DataFrame row) for prediction
        # Model expects a 2D array/DataFrame
        input_array = np.array(input_data_list).reshape(1, -1) # Reshape for single prediction

        # Make prediction
        prediction = model.predict(input_array)
        predicted_stress_level = int(prediction[0]) # Get the scalar prediction

        # Map numerical prediction to a human-readable label
        stress_level_map = {0: "Low Stress", 1: "Moderate Stress", 2: "High Stress"}
        stress_label = stress_level_map.get(predicted_stress_level, "Unknown Stress Level")

        return jsonify({
            'prediction': predicted_stress_level,
            'stress_label': stress_label
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500

# --- Main execution ---
if __name__ == '__main__':
    # Set debug=True for development, it will automatically reload the server
    # when you save changes to the Python files.
    # For production, set debug=False and use a proper WSGI server like Gunicorn.
    app.run(debug=True)