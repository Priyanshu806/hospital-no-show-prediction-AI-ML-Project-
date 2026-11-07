"""
generate_data.py - Synthetic Dataset Generator
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This script creates a reproducible synthetic dataset of hospital appointments.
All features represent information known strictly at the time of booking.
We use random seed = 42 so that running this script always gives the exact same data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_appointment_dataset(total_records=3500, random_seed=42):
    """
    Generates a realistic synthetic dataset of hospital appointments.
    
    Parameters:
        total_records (int): Number of records to create (default is 3,500).
        random_seed (int): Seed for reproducibility (default is 42).
        
    Returns:
        pd.DataFrame: Clean dataset sorted chronologically by booking date.
    """
    # Setting the seed ensures that all random numbers are identical every time we run.
    np.random.seed(random_seed)

    print(f"Generating {total_records} appointment records with seed={random_seed}...")

    # -------------------------------------------------------------
    # Step 1: Appointment Identifiers
    # -------------------------------------------------------------
    # Create simple readable IDs like APT-10000, APT-10001, etc.
    appointment_ids = []
    for i in range(total_records):
        appointment_ids.append(f"APT-{10000 + i}")

    # -------------------------------------------------------------
    # Step 2: Booking Dates and Lead Times
    # -------------------------------------------------------------
    # Booking dates range across 6 months: Jan 1, 2025 to Jun 30, 2025
    start_date = datetime(2025, 1, 1)
    
    booking_dates = []
    appointment_dates = []
    lead_times = []

    # Lead time is the number of days between booking and the actual appointment.
    # Most appointments are scheduled 0 to 14 days ahead, but some are up to 45 days.
    raw_lead_days = np.random.exponential(scale=10.0, size=total_records).astype(int)
    # Enforce realistic hospital lead time between 0 and 45 days
    clean_lead_days = np.clip(raw_lead_days, 0, 45)

    for i in range(total_records):
        # Pick a random day in the 180-day booking window
        day_offset = int(np.random.randint(0, 181))
        b_date = start_date + timedelta(days=day_offset)
        
        lead_day = int(clean_lead_days[i])
        # Appointment date is strictly on or after the booking date
        a_date = b_date + timedelta(days=lead_day)
        
        booking_dates.append(b_date.strftime("%Y-%m-%d"))
        appointment_dates.append(a_date.strftime("%Y-%m-%d"))
        lead_times.append(lead_day)

    lead_times = np.array(lead_times)

    # -------------------------------------------------------------
    # Step 3: Categorical Features (Known at Booking Time)
    # -------------------------------------------------------------
    # Patient age category
    age_categories = [
        "Young Adults (18-35)",
        "Middle-Aged (36-55)",
        "Older Adults (56-70)",
        "Seniors (71+)"
    ]
    age_probabilities = [0.30, 0.35, 0.23, 0.12]
    age_groups = np.random.choice(age_categories, size=total_records, p=age_probabilities)

    # Hospital department
    department_list = [
        "General Medicine",
        "Cardiology",
        "Orthopedics",
        "Dermatology",
        "Pediatrics",
        "Ophthalmology"
    ]
    department_probabilities = [0.30, 0.15, 0.15, 0.15, 0.13, 0.12]
    departments = np.random.choice(department_list, size=total_records, p=department_probabilities)

    # Appointment visit type
    type_list = [
        "Routine Checkup",
        "Follow-up",
        "Specialist Consultation",
        "Diagnostic / Lab"
    ]
    type_probabilities = [0.35, 0.30, 0.20, 0.15]
    appointment_types = np.random.choice(type_list, size=total_records, p=type_probabilities)

    # Booking channel
    channel_list = ["Online Portal", "Phone Call", "In-Person", "Mobile App"]
    channel_probabilities = [0.35, 0.30, 0.15, 0.20]
    booking_channels = np.random.choice(channel_list, size=total_records, p=channel_probabilities)

    # SMS notification opt-in (1 = Yes, 0 = No)
    sms_opt_in = np.random.choice([1, 0], size=total_records, p=[0.65, 0.35])

    # -------------------------------------------------------------
    # Step 4: Prior Appointment History (Known Before Booking)
    # -------------------------------------------------------------
    # Number of prior hospital visits (0 to 15)
    previous_appointments = np.random.poisson(lam=3.0, size=total_records)
    previous_appointments = np.clip(previous_appointments, 0, 15)

    # Number of prior missed appointments (must be <= previous_appointments)
    previous_no_shows = []
    for prev_count in previous_appointments:
        if prev_count == 0:
            previous_no_shows.append(0)
        else:
            # Patient's historical cancellation tendency
            historical_rate = np.random.beta(a=1.5, b=6.0)
            missed_count = np.random.binomial(n=prev_count, p=historical_rate)
            previous_no_shows.append(int(missed_count))
            
    previous_no_shows = np.array(previous_no_shows)

    # -------------------------------------------------------------
    # Step 5: Realistic No-Show Probability & Target Generation
    # -------------------------------------------------------------
    # Base hospital no-show probability is around 14%
    base_probability = 0.14

    # Effect 1: Longer advance lead time increases forgetting (+0.007 per day)
    lead_effect = lead_times * 0.007

    # Effect 2: Prior missed appointments strongly increase risk
    prior_ratio = np.zeros(total_records, dtype=float)
    has_history = previous_appointments > 0
    prior_ratio[has_history] = previous_no_shows[has_history] / previous_appointments[has_history]
    history_effect = (previous_no_shows * 0.08) + (prior_ratio * 0.15)

    # Effect 3: SMS reminder opt-in reduces risk
    sms_effect = np.where(sms_opt_in == 1, -0.06, 0.03)

    # Effect 4: Booking channel effect (In-person shows commitment, Online is easier to miss)
    channel_effect = np.zeros(total_records)
    channel_effect[booking_channels == "Online Portal"] = 0.03
    channel_effect[booking_channels == "In-Person"] = -0.05
    channel_effect[booking_channels == "Mobile App"] = 0.01

    # Effect 5: Age group effect (Younger adults have higher mobility/work conflicts)
    age_effect = np.zeros(total_records)
    age_effect[age_groups == "Young Adults (18-35)"] = 0.04
    age_effect[age_groups == "Seniors (71+)"] = -0.04

    # Effect 6: Department effect (Cardiology is higher perceived urgency than Dermatology)
    department_effect = np.zeros(total_records)
    department_effect[departments == "Cardiology"] = -0.05
    department_effect[departments == "Dermatology"] = 0.03
    department_effect[departments == "General Medicine"] = 0.02

    # Effect 7: Small random real-world noise
    random_noise = np.random.normal(loc=0.0, scale=0.04, size=total_records)

    # Calculate final probability for each appointment
    total_risk = (
        base_probability
        + lead_effect
        + history_effect
        + sms_effect
        + channel_effect
        + age_effect
        + department_effect
        + random_noise
    )

    # Keep probabilities within realistic boundaries [2%, 95%]
    final_probabilities = np.clip(total_risk, 0.02, 0.95)

    # Generate binary target: 1 = No-Show (Missed), 0 = Attended
    random_draw = np.random.rand(total_records)
    no_show = (random_draw < final_probabilities).astype(int)

    # -------------------------------------------------------------
    # Step 6: Create and Sort DataFrame
    # -------------------------------------------------------------
    dataset = pd.DataFrame({
        "appointment_id": appointment_ids,
        "booking_date": booking_dates,
        "appointment_date": appointment_dates,
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

    # Sort chronologically by booking date to prepare for temporal splitting
    dataset["temp_date"] = pd.to_datetime(dataset["booking_date"])
    dataset = dataset.sort_values(by=["temp_date", "appointment_id"]).reset_index(drop=True)
    dataset = dataset.drop(columns=["temp_date"])

    return dataset


def main():
    """Generates and saves the synthetic dataset to data/appointments.csv."""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_directory, "data")
    os.makedirs(output_folder, exist_ok=True)
    
    file_path = os.path.join(output_folder, "appointments.csv")

    df = generate_appointment_dataset(total_records=3500, random_seed=42)
    df.to_csv(file_path, index=False)

    print("==================================================")
    print("Dataset Generation Complete!")
    print(f"Saved to       : {file_path}")
    print(f"Total Rows     : {len(df)}")
    print(f"Total Columns  : {len(df.columns)}")
    print("Class Balance  :")
    print(df["no_show"].value_counts(normalize=True).round(4) * 100)
    print("==================================================")


if __name__ == "__main__":
    main()
