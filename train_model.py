#!/usr/bin/env python3
# train_model.py - Advanced Model Training with Hyperparameter Tuning

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = 'data/stress_data.csv'
MODEL_DIR = 'models'
MODEL_FILE = 'stress_model.pkl'
SCALER_FILE = 'stress_scaler.pkl'

# IMPORTANT: Ensure these column names match your CSV headers exactly.
FEATURE_NAMES = [
    'sleep_quality',
    'exercise_minutes',
    'hours_worked',
    'social_interactions',
    'work_load_score',
    'heart_rate'
]
TARGET_NAME = 'stress_level'

# Advanced Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCALE_FEATURES = True
TUNE_HYPERPARAMETERS = True

def load_and_prepare_data():
    """Load and prepare data with validation."""
    print("=" * 60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 60)
    
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✓ Successfully loaded dataset from '{DATA_FILE}'")
        print(f"  Dataset shape: {df.shape} (rows, columns)")
    except FileNotFoundError:
        print(f"✗ Dataset file not found at '{DATA_FILE}'")
        return None, None, None, None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None, None, None, None

    # Validate columns
    all_required_columns = FEATURE_NAMES + [TARGET_NAME]
    missing_cols = [col for col in all_required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        print(f"  Available columns: {df.columns.tolist()}")
        return None, None, None, None

    # Handle missing values
    print(f"\n  Checking for missing values...")
    missing_values = df[all_required_columns].isnull().sum()
    if missing_values.sum() > 0:
        print(f"  Found missing values, dropping rows...")
        df = df.dropna(subset=all_required_columns)
        print(f"  New shape after dropping NaN: {df.shape}")
    
    X = df[FEATURE_NAMES].astype(float)
    y = df[TARGET_NAME]
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Class distribution:\n{y.value_counts()}")
    
    return X, y, df, all_required_columns

def split_and_scale_data(X, y):
    """Split data and apply feature scaling."""
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING AND SCALING DATA")
    print("=" * 60)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"✓ Data split complete:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
    except ValueError as ve:
        print(f"✗ Error during train-test split: {ve}")
        return None, None, None, None, None
    
    # Feature Scaling
    scaler = None
    if SCALE_FEATURES:
        print(f"\n  Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"✓ Features scaled successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test, None

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n✓ {model_name} Evaluation:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'name': model_name
    }

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with hyperparameter tuning."""
    print("\n" + "-" * 60)
    print("Model 1: Logistic Regression")
    print("-" * 60)
    
    if TUNE_HYPERPARAMETERS:
        # Be mindful of solver compatibility with multiclass; use multinomial for multiclass.
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000, 2000],
            'multi_class': ['multinomial']
        }
        
        lr = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)
        # Use a stratified split for CV to preserve class distribution
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            lr, param_grid, cv=cv,
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
    else:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
        model.fit(X_train, y_train)
    
    return model, evaluate_model(model, X_test, y_test, "Logistic Regression")

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with hyperparameter tuning."""
    print("\n" + "-" * 60)
    print("Model 2: Random Forest Classifier")
    print("-" * 60)
    
    if TUNE_HYPERPARAMETERS:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv,
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
    
    return model, evaluate_model(model, X_test, y_test, "Random Forest")

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting with hyperparameter tuning."""
    print("\n" + "-" * 60)
    print("Model 3: Gradient Boosting Classifier")
    print("-" * 60)
    
    if TUNE_HYPERPARAMETERS:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            gb, param_grid, cv=cv,
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
    else:
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
    
    return model, evaluate_model(model, X_test, y_test, "Gradient Boosting")

def save_model(model, scaler):
    """Save trained model and scaler."""
    print("\n" + "=" * 60)
    print("STEP 3: SAVING MODEL AND SCALER")
    print("=" * 60)
    
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save model
        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILE))
        print(f"✓ Model saved to '{os.path.join(MODEL_DIR, MODEL_FILE)}'")
        
        # Save scaler if used
        if scaler is not None:
            joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILE))
            print(f"✓ Scaler saved to '{os.path.join(MODEL_DIR, SCALER_FILE)}'")
            
    except Exception as e:
        print(f"✗ Error saving model: {e}")

def main():
    """Main training pipeline."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ADVANCED STRESS PREDICTION MODEL TRAINING" + " " * 7 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Load data
    X, y, df, _ = load_and_prepare_data()
    if X is None:
        return
    
    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    if X_train is None:
        return
    
    # Train multiple models
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MODELS WITH ADVANCED SETTINGS")
    print("=" * 60)
    
    models_results = []
    
    # Logistic Regression
    lr_model, lr_results = train_logistic_regression(X_train, X_test, y_train, y_test)
    models_results.append(lr_results)
    
    # Random Forest
    rf_model, rf_results = train_random_forest(X_train, X_test, y_train, y_test)
    models_results.append(rf_results)
    
    # Gradient Boosting
    gb_model, gb_results = train_gradient_boosting(X_train, X_test, y_train, y_test)
    models_results.append(gb_results)
    
    # Select best model
    print("\n" + "=" * 60)
    print("STEP 4: MODEL COMPARISON AND SELECTION")
    print("=" * 60)
    
    best_result = max(models_results, key=lambda x: x['f1'])
    print(f"\n✓ Best Model: {best_result['name']}")
    print(f"  F1-Score: {best_result['f1']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    
    # Save best model
    save_model(best_result['model'], scaler)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - ADVANCED MODEL READY")
    print("=" * 60)
    print(f"\n✓ Features used: {FEATURE_NAMES}")
    print(f"✓ Feature scaling: {'Enabled' if SCALE_FEATURES else 'Disabled'}")
    print(f"✓ Hyperparameter tuning: {'Enabled' if TUNE_HYPERPARAMETERS else 'Disabled'}")
    print(f"✓ Cross-validation folds: {CV_FOLDS}")
    print("\nModel is ready for deployment in app.py!\n")

if __name__ == '__main__':
    main()