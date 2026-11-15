"""
test_pipeline.py - Automated Unit & Integration Tests
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This test suite verifies:
1. Dataset file exists and has >= 3,000 rows.
2. All 12 required columns are present with valid schema.
3. No invalid negative lead times exist in the dataset.
4. Saved model artifact (best_model.joblib) loads and contains pipeline and threshold.
5. Preprocessing pipeline correctly scales numbers and encodes categories without NaNs.
6. Model output probabilities are strictly bounded in [0.0, 1.0].
7. End-to-end sample inference produces valid types and recommendations.
8. Model predictions are consistent and deterministic across repeated calls.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from datetime import date

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "appointments.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")


def test_01_dataset_exists_and_meets_record_count():
    """Verify that the synthetic dataset exists and meets the >= 3,000 records requirement."""
    assert os.path.exists(DATA_PATH), f"Dataset file not found at: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 3000, f"Expected at least 3,000 records, but found {len(df)}"


def test_02_dataset_has_all_required_columns():
    """Verify that all 12 required features and target columns are present."""
    df = pd.read_csv(DATA_PATH)
    expected_columns = [
        "appointment_id",
        "booking_date",
        "appointment_date",
        "age_group",
        "appointment_type",
        "department",
        "booking_channel",
        "previous_appointments",
        "previous_no_shows",
        "sms_opt_in",
        "lead_time_days",
        "no_show"
    ]
    for column_name in expected_columns:
        assert column_name in df.columns, f"Required column '{column_name}' missing from dataset"


def test_03_no_negative_lead_times_in_dataset():
    """Verify that no appointment has an invalid date (appointment before booking date)."""
    df = pd.read_csv(DATA_PATH)
    negative_lead_times = (df["lead_time_days"] < 0).sum()
    assert negative_lead_times == 0, f"Found {negative_lead_times} negative lead time records"


def test_04_saved_model_artifact_contains_required_keys():
    """Verify that the saved model artifact loads cleanly and contains pipeline and threshold."""
    assert os.path.exists(MODEL_PATH), f"Model artifact not found at: {MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)

    # Check required dictionary keys
    assert "pipeline" in artifact, "Artifact missing 'pipeline' key"
    assert "threshold" in artifact, "Artifact missing 'threshold' key"
    assert "feature_names" in artifact, "Artifact missing 'feature_names' key"
    assert "model_name" in artifact, "Artifact missing 'model_name' key"

    # Verify operating threshold is valid
    assert 0.0 < artifact["threshold"] < 1.0, f"Threshold {artifact['threshold']} is out of valid range"


def test_05_preprocessing_pipeline_handles_sample_input():
    """Verify that the preprocessor correctly transforms numerical and categorical data without NaNs."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]

    sample_appointment = pd.DataFrame([{
        "lead_time_days": 10,
        "previous_appointments": 3,
        "previous_no_shows": 1,
        "age_group": "Middle-Aged (36-55)",
        "appointment_type": "Specialist Consultation",
        "department": "Cardiology",
        "booking_channel": "Online Portal",
        "sms_opt_in": 1
    }])

    transformed_data = preprocessor.transform(sample_appointment)
    assert not np.isnan(transformed_data).any(), "Transformed feature matrix contains unexpected NaNs"
    assert transformed_data.shape[1] > 0, "Transformed feature matrix has zero columns"


def test_06_predicted_probabilities_are_between_0_and_1():
    """Verify that predict_proba returns probabilities strictly bounded between 0.0 and 1.0."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    test_samples = pd.DataFrame([
        {
            "lead_time_days": 1,
            "previous_appointments": 5,
            "previous_no_shows": 0,
            "age_group": "Seniors (71+)",
            "appointment_type": "Follow-up",
            "department": "Cardiology",
            "booking_channel": "In-Person",
            "sms_opt_in": 1
        },
        {
            "lead_time_days": 35,
            "previous_appointments": 4,
            "previous_no_shows": 3,
            "age_group": "Young Adults (18-35)",
            "appointment_type": "Routine Checkup",
            "department": "Dermatology",
            "booking_channel": "Online Portal",
            "sms_opt_in": 0
        }
    ])

    probabilities = pipeline.predict_proba(test_samples)[:, 1]
    for prob in probabilities:
        assert 0.0 <= prob <= 1.0, f"Probability {prob} is outside valid range [0, 1]"


def test_07_sample_prediction_formats_and_types():
    """Verify end-to-end inference on a single representative patient appointment."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    sample_patient = pd.DataFrame([{
        "lead_time_days": 14,
        "previous_appointments": 2,
        "previous_no_shows": 0,
        "age_group": "Middle-Aged (36-55)",
        "appointment_type": "Routine Checkup",
        "department": "General Medicine",
        "booking_channel": "Phone Call",
        "sms_opt_in": 1
    }])

    prob = float(pipeline.predict_proba(sample_patient)[0, 1])
    is_suggested = bool(prob >= threshold)

    assert isinstance(prob, float), "Predicted probability must be a float"
    assert isinstance(is_suggested, bool), "Decision flag must be a boolean"
    assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"


def test_08_prediction_is_deterministic_and_consistent():
    """Verify that calling predict_proba multiple times on identical input yields the exact same score."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    sample_patient = pd.DataFrame([{
        "lead_time_days": 8,
        "previous_appointments": 2,
        "previous_no_shows": 1,
        "age_group": "Young Adults (18-35)",
        "appointment_type": "Follow-up",
        "department": "Orthopedics",
        "booking_channel": "Mobile App",
        "sms_opt_in": 0
    }])

    prob_1 = pipeline.predict_proba(sample_patient)[0, 1]
    prob_2 = pipeline.predict_proba(sample_patient)[0, 1]
    prob_3 = pipeline.predict_proba(sample_patient)[0, 1]

    assert prob_1 == prob_2 == prob_3, "Predictions are non-deterministic for identical inputs"
