"""
predict.py - Command-Line Prediction Tool
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

Usage:
    python predict.py
    python predict.py --lead_time 21 --prev_appts 3 --prev_noshows 2 --dept Dermatology
"""

import os
import sys
import argparse
import warnings

# Suppress version and deprecation warnings for clean student output
warnings.filterwarnings("ignore")

import joblib
import pandas as pd


def get_model_path():
    """Finds the model artifact whether running from workspace root or project subfolder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "models", "best_model.joblib"),
        os.path.join(current_dir, "hospital-no-show-prediction", "models", "best_model.joblib"),
        os.path.join("models", "best_model.joblib"),
        os.path.join("hospital-no-show-prediction", "models", "best_model.joblib")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_pipeline():
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}\nPlease run train_model.py first.")
    return joblib.load(model_path)


def predict_sample(lead_time_days=14,
                   previous_appointments=2,
                   previous_no_shows=1,
                   age_group="Young Adults (18-35)",
                   appointment_type="Routine Checkup",
                   department="General Medicine",
                   booking_channel="Online Portal",
                   sms_opt_in=1):
    artifact = load_pipeline()
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    input_df = pd.DataFrame([{
        "lead_time_days": lead_time_days,
        "previous_appointments": previous_appointments,
        "previous_no_shows": previous_no_shows,
        "age_group": age_group,
        "appointment_type": appointment_type,
        "department": department,
        "booking_channel": booking_channel,
        "sms_opt_in": sms_opt_in
    }])

    prob = float(pipeline.predict_proba(input_df)[0, 1])
    is_suggested = prob >= threshold

    print("\n" + "=" * 60)
    print("🏥 HOSPITAL APPOINTMENT NO-SHOW PREDICTION (CLI INFERENCE)")
    print("=" * 60)
    print(f"[*] Input Features:")
    print(f"    - Lead Time            : {lead_time_days} days in advance")
    print(f"    - Historical Attendance: {previous_appointments} previous ({previous_no_shows} missed)")
    print(f"    - Patient Demographics : {age_group}")
    print(f"    - Clinic Department    : {department} ({appointment_type})")
    print(f"    - Booking Medium       : {booking_channel}")
    print(f"    - SMS Opt-in Status    : {'Yes' if sms_opt_in == 1 else 'No'}")
    print("-" * 60)
    print(f"[+] Predicted No-Show Probability : {prob * 100:.1f}%")
    print(f"[+] Operational Decision Threshold: {threshold * 100:.0f}%")
    if is_suggested:
        print(f"[!] RECOMMENDATION                : ⚠️  Reminder Review: Suggested")
        print("    Action: Flagged for staff priority reminder call (24-48 hours before).")
    else:
        print(f"[!] RECOMMENDATION                : ✅  Reminder Review: Not Prioritized")
        print("    Action: Standard automated reminders sufficient. Low no-show risk.")
    print("-" * 60)
    print("Disclaimer: Model output is advisory decision-support for human staff review.")
    print("=" * 60 + "\n")
    return prob, is_suggested


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hospital Appointment No-Show Prediction CLI")
    parser.add_argument("--lead_time", type=int, default=14, help="Advance booking lead time in days")
    parser.add_argument("--prev_appts", type=int, default=2, help="Total prior appointments")
    parser.add_argument("--prev_noshows", type=int, default=1, help="Total prior missed visits")
    parser.add_argument("--age_group", type=str, default="Young Adults (18-35)", help="Age group category")
    parser.add_argument("--type", type=str, default="Routine Checkup", help="Appointment visit type")
    parser.add_argument("--dept", type=str, default="General Medicine", help="Clinical department")
    parser.add_argument("--channel", type=str, default="Online Portal", help="Booking channel")
    parser.add_argument("--sms", type=int, default=1, help="SMS opt-in status (1=Yes, 0=No)")

    args = parser.parse_args()
    predict_sample(
        lead_time_days=args.lead_time,
        previous_appointments=args.prev_appts,
        previous_no_shows=args.prev_noshows,
        age_group=args.age_group,
        appointment_type=args.type,
        department=args.dept,
        booking_channel=args.channel,
        sms_opt_in=args.sms
    )
