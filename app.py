# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib  # For loading the scikit-learn model

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('models/stress_model.pkl')
    # Assuming your model expects features in a specific order
    feature_names = [
        'sleep_quality',
        'exercise_minutes',
        'hours_worked',
        'social_interactions',
        'work_load_score',
        'heart_rate' # <-- Added heart_rate here
    ]
except FileNotFoundError:
    model = None
    feature_names = []
    print("Error: Model file not found. Please train and save your model.")
except Exception as e:
    model = None
    feature_names = []
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        # Extract features from the JSON data and ensure correct order
        # IMPORTANT: Replace 'feature1', 'feature2', etc. with your actual feature names
        # and ensure the order matches what your model was trained on.
        input_features = [data.get(feature) for feature in feature_names]

        # Check if all required features are present
        if None in input_features:
            missing_features = [feature_names[i] for i, val in enumerate(input_features) if val is None]
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400

        # Convert to DataFrame suitable for the model
        input_df = pd.DataFrame([input_features], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)
        prediction_value = int(prediction[0]) # Assuming a single prediction value

        # You might want to map the prediction value to a more user-friendly string
        # Example: if prediction_value == 1, it means 'High Stress'
        # stress_level = "High Stress" if prediction_value == 1 else "Low Stress"

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    # Debug=True is useful for development, but set to False for production
    app.run(debug=True)