"""
generate_data.py - Synthetic Dataset Generator for Hospital Appointment No-Show Prediction
Project: Hospital Appointment No-Show Prediction
Author: B.Tech Computer Science & Engineering (Data Science / ML)
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This script creates a realistic, reproducible synthetic dataset of hospital appointments.
All features strictly represent information available at booking time.
Random Seed: 42 (Guarantees exact reproducibility).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_appointment_dataset(num_records: int = 3500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a reproducible synthetic dataset of hospital appointments.

    Parameters:
        num_records (int): Total number of records to generate (default: 3500 >= 3000 requirement).
        random_seed (int): Seed for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Synthetic dataset sorted chronologically by booking_date.
    """
    np.random.seed(random_seed)

    # 1. Generate Appointment IDs
    appointment_ids = [f"APT-{10000 + i}" for i in range(num_records)]

    # 2. Generate Booking Dates over a 6-month timeline (e.g., Jan 1, 2025 to Jun 30, 2025)
    start_date = datetime(2025, 1, 1)
    booking_offsets = np.random.randint(0, 181, size=num_records)  # up to 180 days
    booking_dates = [start_date + timedelta(days=int(offset)) for offset in booking_offsets]

    # 3. Generate Lead Time in Days (Days between booking and appointment)
    # Using an exponential-like distribution: most appointments are within 0 to 45 days
    # Same-day or next-day appointments are common, but some are scheduled weeks ahead.
    lead_times = np.random.exponential(scale=10.0, size=num_records).astype(int)
    lead_times = np.clip(lead_times, 0, 45)  # Enforce reasonable hospital scheduling range (0 to 45 days)

    # Calculate Appointment Dates: appointment_date = booking_date + lead_time_days
    # Ensures appointment_date is never before booking_date.
    appointment_dates = [
        b_date + timedelta(days=int(l_days))
        for b_date, l_days in zip(booking_dates, lead_times)
    ]

    # 4. Generate Categorical Patient & Appointment Features (Available at Booking Time)
    age_groups_list = [
        "Young Adults (18-35)",
        "Middle-Aged (36-55)",
        "Older Adults (56-70)",
        "Seniors (71+)"
    ]
    age_group_probs = [0.30, 0.35, 0.23, 0.12]
    age_groups = np.random.choice(age_groups_list, size=num_records, p=age_group_probs)

    departments_list = [
        "General Medicine",
        "Cardiology",
        "Orthopedics",
        "Dermatology",
        "Pediatrics",
        "Ophthalmology"
    ]
    dept_probs = [0.30, 0.15, 0.15, 0.15, 0.13, 0.12]
    departments = np.random.choice(departments_list, size=num_records, p=dept_probs)

    appointment_types_list = [
        "Routine Checkup",
        "Follow-up",
        "Specialist Consultation",
        "Diagnostic / Lab"
    ]
    type_probs = [0.35, 0.30, 0.20, 0.15]
    appointment_types = np.random.choice(appointment_types_list, size=num_records, p=type_probs)

    booking_channels_list = ["Online Portal", "Phone Call", "In-Person", "Mobile App"]
    channel_probs = [0.35, 0.30, 0.15, 0.20]
    booking_channels = np.random.choice(booking_channels_list, size=num_records, p=channel_probs)

    # SMS Opt-in (1 = Opted in, 0 = Not opted in)
    sms_opt_in = np.random.choice([1, 0], size=num_records, p=[0.65, 0.35])

    # 5. Generate Patient Prior History (strictly known before booking time)
    # Poisson distribution for historical appointments count
    previous_appointments = np.random.poisson(lam=3.0, size=num_records)
    previous_appointments = np.clip(previous_appointments, 0, 15)

    # Previous no-shows must be <= previous_appointments
    previous_no_shows = []
    for prev_count in previous_appointments:
        if prev_count == 0:
            previous_no_shows.append(0)
        else:
            # Historical no-show probability for an individual is drawn from a beta distribution
            hist_rate = np.random.beta(a=1.5, b=6.0)
            no_show_count = np.random.binomial(n=prev_count, p=hist_rate)
            previous_no_shows.append(int(no_show_count))
    previous_no_shows = np.array(previous_no_shows)

    # 6. Generate Realistic Target Variable: no_show (0 = Attended, 1 = Missed)
    # Documented synthetic logic based on operational healthcare assumptions:
    # - Base baseline no-show rate: ~14%
    # - Longer lead time increases forgetting rate (+0.007 per day)
    # - Prior no-show behavior strongly correlates with future no-shows (+0.12 per prior no-show)
    # - SMS notification opt-in reduces no-show risk (-0.06)
    # - Booking channel: In-person shows commitment (-0.05), Online Portal has less friction (+0.03)
    # - Age groups: Young Adults have slightly higher no-show (+0.04), Seniors lower (-0.04)
    # - Department: Cardiology/Critical lower (-0.05), Routine Dermatology higher (+0.03)
    base_prob = 0.14
    lead_effect = lead_times * 0.007
    
    # Previous no-show ratio effect
    prev_ratio = np.zeros(num_records, dtype=float)
    valid_mask = previous_appointments > 0
    prev_ratio[valid_mask] = previous_no_shows[valid_mask] / previous_appointments[valid_mask]
    history_effect = (previous_no_shows * 0.08) + (prev_ratio * 0.15)
    
    sms_effect = np.where(sms_opt_in == 1, -0.06, 0.03)
    
    channel_effect = np.zeros(num_records)
    channel_effect[booking_channels == "Online Portal"] = 0.03
    channel_effect[booking_channels == "In-Person"] = -0.05
    channel_effect[booking_channels == "Mobile App"] = 0.01

    age_effect = np.zeros(num_records)
    age_effect[age_groups == "Young Adults (18-35)"] = 0.04
    age_effect[age_groups == "Seniors (71+)"] = -0.04

    dept_effect = np.zeros(num_records)
    dept_effect[departments == "Cardiology"] = -0.05
    dept_effect[departments == "Dermatology"] = 0.03
    dept_effect[departments == "General Medicine"] = 0.02

    # Add controlled random variation (stochastic noise)
    noise = np.random.normal(loc=0.0, scale=0.04, size=num_records)

    # Calculate overall probability
    calculated_probs = (
        base_prob
        + lead_effect
        + history_effect
        + sms_effect
        + channel_effect
        + age_effect
        + dept_effect
        + noise
    )

    # Clip probabilities between 2% and 95%
    clipped_probs = np.clip(calculated_probs, 0.02, 0.95)

    # Determine binary outcome: 1 = no-show (missed), 0 = attended
    no_show = (np.random.rand(num_records) < clipped_probs).astype(int)

    # 7. Construct DataFrame
    df = pd.DataFrame({
        "appointment_id": appointment_ids,
        "booking_date": [d.strftime("%Y-%m-%d") for d in booking_dates],
        "appointment_date": [d.strftime("%Y-%m-%d") for d in appointment_dates],
        "age_group": age_groups,
        "appointment_type": appointment_types,
        "department": departments,
        "booking_channel": booking_channels,
        "previous_appointments": previous_appointments,
        "previous_no_shows": previous_no_shows,
        "sms_opt_in": sms_opt_in,
        "lead_time_days": lead_times,
        "no_show": no_show
    })

    # Sort chronologically by booking_date to prevent future leakage in time-based splitting
    df["booking_date_dt"] = pd.to_datetime(df["booking_date"])
    df = df.sort_values(by=["booking_date_dt", "appointment_id"]).reset_index(drop=True)
    df = df.drop(columns=["booking_date_dt"])

    return df


def main():
    """Generates and saves the synthetic dataset to data/appointments.csv."""
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "appointments.csv")

    print("[*] Generating synthetic hospital appointments dataset...")
    df = generate_appointment_dataset(num_records=3500, random_seed=42)

    df.to_csv(output_path, index=False)
    print(f"[+] Dataset successfully created at: {output_path}")
    print(f"[i] Total records: {len(df)}")
    print(f"[i] Columns: {list(df.columns)}")
    print(f"[i] Target distribution:\n{df['no_show'].value_counts(normalize=True).round(4) * 100}%")
    print(f"[i] Booking date range: {df['booking_date'].min()} to {df['booking_date'].max()}")
    print(f"[i] Lead time range: {df['lead_time_days'].min()} to {df['lead_time_days'].max()} days")


if __name__ == "__main__":
    main()
