# train_model.py
# This script should be saved in the root directory of your project.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Example model
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os # Import os module for path manipulation

# --- Configuration ---
DATA_FILE = 'data/stress_data.csv'
MODEL_DIR = 'models'
MODEL_FILE = 'stress_model.pkl'

# Define features (X) and target (y)
# IMPORTANT: Ensure these names exactly match your CSV column headers!
FEATURE_NAMES = [
    'sleep_quality',
    'exercise_minutes',
    'hours_worked',
    'social_interactions',
    'work_load_score',
    'heart_rate'
]
TARGET_NAME = 'stress_level'

# --- Script Execution ---

def train_and_save_model():
    """
    Loads data, trains a stress prediction model, and saves it.
    """
    print("Starting model training process...")

    # 1. Load your dataset
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Successfully loaded dataset from '{DATA_FILE}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{DATA_FILE}'.")
        print("Please ensure 'data/stress_data.csv' exists and contains your data.")
        return # Exit function if data file is missing
    except Exception as e:
        print(f"An unexpected error occurred while loading '{DATA_FILE}': {e}")
        return

    # 2. Data Preprocessing and Feature Engineering (Placeholder)
    # IMPORTANT: Add your actual preprocessing steps here if needed.
    # For this example, we assume the data is largely ready or simple transformations suffice.
    # Example:
    # df['encoded_feature'] = df['categorical_column'].astype('category').cat.codes

    print("Performing basic checks on columns...")
    # 3. Define features (X) and target (y) and validate columns
    all_required_columns = FEATURE_NAMES + [TARGET_NAME]
    missing_cols = [col for col in all_required_columns if col not in df.columns]

    if missing_cols:
        print(f"Error: The following required columns are missing from '{DATA_FILE}': {missing_cols}")
        print("Please check your CSV file and ensure all column names match the FEATURE_NAMES and TARGET_NAME variables.")
        return

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    print("Columns validated. Features and target defined.")

    # 4. Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    # Using stratify=y is recommended for classification to maintain class proportions
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,  # 20% for testing
            random_state=42, # for reproducibility
            stratify=y      # ensures class distribution is similar in train/test sets
        )
        print(f"Data split complete. Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    except ValueError as ve:
        print(f"Error during train-test split: {ve}")
        print("This might happen if your target variable ('stress_level') has too few samples per class.")
        print("Consider reducing test_size, combining classes, or gathering more data.")
        return

    # 5. Model Training
    print("Initializing and training the Logistic Regression model...")
    # Logistic Regression is a good baseline. Choose a model that suits your data complexity.
    # Higher max_iter might be needed for convergence on some datasets.
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs') # Using liblinear for smaller datasets, robust

    try:
        model.fit(X_train, y_train)
        print("Model training completed successfully.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return

    # 6. Model Evaluation (Optional but Recommended)
    print("\n--- Model Evaluation ---")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on the test set: {accuracy:.4f}")

        # Detailed report for classification
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0)) # zero_division handles classes with no predictions
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")

    # 7. Save the trained model
    print(f"\nSaving the trained model to '{os.path.join(MODEL_DIR, MODEL_FILE)}'...")
    try:
        # Create the models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Use joblib to save the model
        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILE))
        print(f"Model successfully saved to '{os.path.join(MODEL_DIR, MODEL_FILE)}'")
    except Exception as e:
        print(f"Error saving model to '{os.path.join(MODEL_DIR, MODEL_FILE)}': {e}")

    # 8. Provide feedback on feature names for Flask
    print("\n--- Next Steps ---")
    print("INFO: The following feature names were used for training:")
    print(FEATURE_NAMES)
    print("Please ensure these exact names are also used in 'app.py' and your HTML form.")
    print("------------------")

if __name__ == '__main__':
    train_and_save_model()