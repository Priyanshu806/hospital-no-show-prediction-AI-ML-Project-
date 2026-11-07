"""
test_pipeline.py - Pytest Verification Suite for Hospital No-Show Prediction
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

Tests cover:
1. Dataset generation & row count
2. Schema & required columns
3. Invalid date & constraint logic
4. Preprocessing pipeline functionality
5. Model artifact loading & integrity
6. Output probability bounds [0, 1]
7. End-to-end sample appointment inference
8. Prediction determinism & consistency
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "appointments.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")


def test_dataset_generation():
    """Verify that the synthetic dataset exists and meets the minimum >= 3,000 records requirement."""
    assert os.path.exists(DATA_PATH), f"Dataset file missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 3000, f"Expected at least 3,000 records, but found {len(df)}"


def test_required_columns():
    """Verify that all required columns are present in the dataset."""
    df = pd.read_csv(DATA_PATH)
    expected_cols = [
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
    for col in expected_cols:
        assert col in df.columns, f"Required column '{col}' is missing from dataset"


def test_invalid_date_handling():
    """Verify that date logic flags or rejects appointment dates that occur before booking dates."""
    booking_date = date(2025, 5, 10)
    invalid_appt_date = date(2025, 5, 5)  # 5 days in past!

    lead_time = (invalid_appt_date - booking_date).days
    assert lead_time < 0, "Lead time should be negative for an invalid past date"

    # Verify that in our dataset, no lead time is negative
    df = pd.read_csv(DATA_PATH)
    negative_lead_times = (df["lead_time_days"] < 0).sum()
    assert negative_lead_times == 0, f"Found {negative_lead_times} negative lead time rows in dataset"


def test_model_artifact_loading():
    """Verify that the saved model artifact exists, loads cleanly, and contains all required keys."""
    assert os.path.exists(MODEL_PATH), f"Model artifact missing at {MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)

    assert "pipeline" in artifact, "Artifact missing 'pipeline' key"
    assert "threshold" in artifact, "Artifact missing 'threshold' key"
    assert "feature_names" in artifact, "Artifact missing 'feature_names' key"
    assert "model_name" in artifact, "Artifact missing 'model_name' key"

    assert 0.0 < artifact["threshold"] < 1.0, f"Threshold {artifact['threshold']} is out of valid range (0, 1)"


def test_preprocessing_pipeline():
    """Verify that the preprocessor in the pipeline correctly transforms raw data without NaN values."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]

    sample_input = pd.DataFrame([{
        "lead_time_days": 12,
        "previous_appointments": 4,
        "previous_no_shows": 1,
        "age_group": "Middle-Aged (36-55)",
        "appointment_type": "Specialist Consultation",
        "department": "Cardiology",
        "booking_channel": "Online Portal",
        "sms_opt_in": 1
    }])

    transformed = preprocessor.transform(sample_input)
    assert not np.isnan(transformed).any(), "Transformed feature matrix contains unexpected NaN values"
    assert transformed.shape[1] > 0, "Transformed feature matrix has zero columns"


def test_prediction_probability_bounds():
    """Verify that the predicted no-show probability is strictly bounded between 0.0 and 1.0."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    test_samples = pd.DataFrame([
        {
            "lead_time_days": 0,
            "previous_appointments": 5,
            "previous_no_shows": 0,
            "age_group": "Seniors (71+)",
            "appointment_type": "Follow-up",
            "department": "Cardiology",
            "booking_channel": "In-Person",
            "sms_opt_in": 1
        },
        {
            "lead_time_days": 40,
            "previous_appointments": 3,
            "previous_no_shows": 3,
            "age_group": "Young Adults (18-35)",
            "appointment_type": "Routine Checkup",
            "department": "Dermatology",
            "booking_channel": "Online Portal",
            "sms_opt_in": 0
        }
    ])

    probabilities = pipeline.predict_proba(test_samples)[:, 1]
    for p in probabilities:
        assert 0.0 <= p <= 1.0, f"Predicted probability {p} is outside [0.0, 1.0]"


def test_valid_prediction_sample():
    """Verify inference on a representative appointment record."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    sample = pd.DataFrame([{
        "lead_time_days": 14,
        "previous_appointments": 2,
        "previous_no_shows": 0,
        "age_group": "Middle-Aged (36-55)",
        "appointment_type": "Routine Checkup",
        "department": "General Medicine",
        "booking_channel": "Phone Call",
        "sms_opt_in": 1
    }])

    prob = pipeline.predict_proba(sample)[0, 1]
    is_suggested = bool(prob >= threshold)

    assert isinstance(prob, float)
    assert isinstance(is_suggested, bool)
    assert 0.0 <= prob <= 1.0


def test_prediction_consistency():
    """Verify that calling predict_proba multiple times on identical input yields deterministic results."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    sample = pd.DataFrame([{
        "lead_time_days": 10,
        "previous_appointments": 3,
        "previous_no_shows": 1,
        "age_group": "Young Adults (18-35)",
        "appointment_type": "Follow-up",
        "department": "Orthopedics",
        "booking_channel": "Mobile App",
        "sms_opt_in": 0
    }])

    prob_1 = pipeline.predict_proba(sample)[0, 1]
    prob_2 = pipeline.predict_proba(sample)[0, 1]
    prob_3 = pipeline.predict_proba(sample)[0, 1]

    assert prob_1 == prob_2 == prob_3, "Predictions are non-deterministic for identical inputs"

