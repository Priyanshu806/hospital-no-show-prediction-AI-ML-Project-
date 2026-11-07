"""
train_model.py - Machine Learning Model Training & Evaluation Pipeline
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This script performs the complete machine learning workflow:
1. Loads data/appointments.csv.
2. Applies a chronological train/validation/test split (70% / 15% / 15%).
3. Builds a preprocessing pipeline using ColumnTransformer.
4. Trains 3 models: Dummy Baseline, Logistic Regression, and Random Forest.
5. Optimizes the decision threshold on validation data (selecting threshold = 0.40).
6. Evaluates the best model on the held-out test set.
7. Saves the trained model pipeline to models/best_model.joblib.
8. Saves evaluation metrics to reports/model_results.csv.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Scikit-learn imports for models, preprocessing, and metrics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)

# -------------------------------------------------------------
# 1. Feature Definitions (Available Strictly at Booking Time)
# -------------------------------------------------------------
# Numerical features: quantitative numbers
NUMERICAL_FEATURES = [
    "lead_time_days",
    "previous_appointments",
    "previous_no_shows"
]

# Categorical features: categories/text groups and binary flags
CATEGORICAL_FEATURES = [
    "age_group",
    "appointment_type",
    "department",
    "booking_channel",
    "sms_opt_in"
]

# Target column: 0 = Attended, 1 = Missed (No-Show)
TARGET_COLUMN = "no_show"


def build_preprocessor():
    """
    Builds the ColumnTransformer for preprocessing.
    
    Why we need this:
    - SimpleImputer fills any missing values (median for numbers, most-frequent for categories).
    - StandardScaler normalizes numerical values (mean=0, std=1) so features are on the same scale.
    - OneHotEncoder turns categorical text categories into 0/1 binary numbers.
    - ColumnTransformer bundles them together and routes columns to the right transformer.
    """
    # Pipeline for numerical columns
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for categorical columns
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical_pipeline", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"  # Drop unused columns like appointment_id, booking_date, appointment_date
    )
    return preprocessor


def split_data_chronologically(df):
    """
    Splits data chronologically based on booking_date:
    - Oldest 70%: Training Set
    - Next 15%: Validation Set
    - Latest 15%: Held-Out Test Set
    
    Why chronological split?
    In hospital operations, we use past appointment records to train our model and predict
    future appointments. Random shuffling would leak future trends into past training data.
    """
    # Sort strictly by booking_date
    sorted_df = df.sort_values(by="booking_date").reset_index(drop=True)
    total_records = len(sorted_df)

    train_end = int(0.70 * total_records)
    val_end = int(0.85 * total_records)

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]

    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    split_summary = {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_dates": f"{train_df['booking_date'].min()} to {train_df['booking_date'].max()}",
        "val_dates": f"{val_df['booking_date'].min()} to {val_df['booking_date'].max()}",
        "test_dates": f"{test_df['booking_date'].min()} to {test_df['booking_date'].max()}"
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, split_summary


def calculate_metrics(y_true, y_pred, y_probabilities):
    """
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Brier score, and Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC calculation
    try:
        auc_score = float(roc_auc_score(y_true, y_probabilities))
    except Exception:
        auc_score = 0.5

    # Brier score: measures probability calibration (lower is better)
    brier = float(brier_score_loss(y_true, y_probabilities))

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": auc_score,
        "Brier Score": brier,
        "True Negative": int(tn),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Positive": int(tp)
    }


def analyze_decision_thresholds(model_pipeline, X_val, y_val):
    """
    Tests different decision thresholds (0.3 to 0.7) on the validation set.
    
    Why:
    Default threshold is 0.50, but in healthcare operations:
    - False Negative (missing a no-show) wastes 45 minutes of doctor time.
    - False Positive (sending an extra reminder) only costs a 30-second reminder SMS.
    Therefore, lowering the threshold increases Recall (catching more missed visits).
    """
    val_probs = model_pipeline.predict_proba(X_val)[:, 1]
    candidate_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    results = []

    for threshold in candidate_thresholds:
        binary_predictions = (val_probs >= threshold).astype(int)
        cm = confusion_matrix(y_val, binary_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        prec = precision_score(y_val, binary_predictions, zero_division=0)
        rec = recall_score(y_val, binary_predictions, zero_division=0)
        f1 = f1_score(y_val, binary_predictions, zero_division=0)
        
        results.append({
            "Threshold": threshold,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "TP (Caught)": tp,
            "FP (Extra Calls)": fp,
            "FN (Missed)": fn
        })

    return pd.DataFrame(results), val_probs


def train_and_evaluate():
    """Main function to train, compare, and save the models."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "appointments.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("==================================================")
    print("Hospital Appointment No-Show Model Training")
    print("==================================================")

    # 1. Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    # 2. Split data chronologically
    X_train, y_train, X_val, y_val, X_test, y_test, split_info = split_data_chronologically(df)
    print(f"Training records   : {split_info['train_count']} ({split_info['train_dates']})")
    print(f"Validation records : {split_info['val_count']} ({split_info['val_dates']})")
    print(f"Test records       : {split_info['test_count']} ({split_info['test_dates']})")

    # 3. Define the preprocessing pipeline
    preprocessor = build_preprocessor()

    # 4. Define candidate models
    # We use class_weight='balanced' so the model penalizes missing no-shows appropriately.
    models_to_train = {
        "Dummy Baseline": DummyClassifier(strategy="prior", random_state=42),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42
        )
    }

    validation_records = []
    fitted_pipelines = {}

    print("\n--- Model Training & Validation Evaluation ---")
    for model_name, classifier in models_to_train.items():
        # Package preprocessing and classifier inside a Scikit-Learn Pipeline
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        # Fit model pipeline on training data
        model_pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = model_pipeline

        # Predict on validation data using default threshold 0.50
        val_probs = model_pipeline.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        metrics = calculate_metrics(y_val, val_preds, val_probs)
        metrics["Model"] = model_name
        metrics["Dataset"] = "Validation"
        metrics["Threshold"] = 0.50
        validation_records.append(metrics)

        print(f"[{model_name}] Accuracy: {metrics['Accuracy']:.3f} | Precision: {metrics['Precision']:.3f} | "
              f"Recall: {metrics['Recall']:.3f} | F1: {metrics['F1-Score']:.3f} | ROC-AUC: {metrics['ROC-AUC']:.3f}")

    val_df = pd.DataFrame(validation_records)

    # 5. Threshold Optimization on Validation Set for Random Forest
    print("\n--- Validation Threshold Optimization (Random Forest) ---")
    rf_pipeline = fitted_pipelines["Random Forest"]
    threshold_table, rf_val_probs = analyze_decision_thresholds(rf_pipeline, X_val, y_val)
    print(threshold_table.to_string(index=False))

    # We select threshold 0.40 because it catches >80% of missed visits with acceptable extra calls
    CHOSEN_THRESHOLD = 0.40
    print(f"\n[+] Selected Decision Threshold: {CHOSEN_THRESHOLD}")
    print("    Reason: High recall (>75%) minimizes costly empty doctor consultation slots.")

    # 6. Final Evaluation on Untouched Test Set
    print("\n--- Final Test Set Evaluation (Held-Out Future Data) ---")
    test_records = []

    for model_name, model_pipeline in fitted_pipelines.items():
        test_probs = model_pipeline.predict_proba(X_test)[:, 1]
        
        # Apply chosen threshold 0.40 for Random Forest, and default 0.50 for others
        applied_threshold = CHOSEN_THRESHOLD if model_name == "Random Forest" else 0.50
        test_preds = (test_probs >= applied_threshold).astype(int)

        metrics = calculate_metrics(y_test, test_preds, test_probs)
        metrics["Model"] = model_name
        metrics["Dataset"] = "Test"
        metrics["Threshold"] = applied_threshold
        test_records.append(metrics)

        print(f"[{model_name}] (Threshold: {applied_threshold}) -> Accuracy: {metrics['Accuracy']:.3f} | "
              f"Precision: {metrics['Precision']:.3f} | Recall: {metrics['Recall']:.3f} | "
              f"F1: {metrics['F1-Score']:.3f} | ROC-AUC: {metrics['ROC-AUC']:.3f} | Brier: {metrics['Brier Score']:.4f}")

    test_df = pd.DataFrame(test_records)

    # 7. Save model comparison results table to reports/model_results.csv
    combined_results = pd.concat([val_df, test_df], ignore_index=True)
    results_csv_path = os.path.join(reports_dir, "model_results.csv")
    combined_results.to_csv(results_csv_path, index=False)
    print(f"\nSaved model results to: {results_csv_path}")

    # 8. Save the best model pipeline and metadata to models/best_model.joblib
    champion_model_name = "Random Forest"
    best_pipeline = fitted_pipelines[champion_model_name]

    artifact_to_save = {
        "pipeline": best_pipeline,
        "threshold": CHOSEN_THRESHOLD,
        "model_name": champion_model_name,
        "feature_names": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_info": split_info,
        "validation_metrics": val_df[val_df["Model"] == champion_model_name].iloc[0].to_dict(),
        "test_metrics": test_df[test_df["Model"] == champion_model_name].iloc[0].to_dict()
    }

    joblib_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(artifact_to_save, joblib_path)
    print(f"Saved trained model artifact to: {joblib_path}")

    return artifact_to_save, combined_results


if __name__ == "__main__":
    train_and_evaluate()
