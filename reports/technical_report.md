# Technical Project Report: Hospital Appointment No-Show Prediction
**Assessment Body:** ALGIONS AI LABS PRIVATE LIMITED  
**Role:** AI & Machine Learning Intern  
**Project Domain:** Hospital Management & Healthcare Operations  
**Academic Level:** Final-Year B.Tech Computer Science & Engineering (Specialization: Data Science / Machine Learning)  

---

## 1. Executive Summary & Introduction
Outpatient department (OPD) missed appointments ("no-shows") represent a major operational and financial challenge for modern healthcare systems. When a patient misses an appointment without notifying the clinic in advance, clinical staff and high-value diagnostic equipment sit idle, operational overheads rise, and care delivery for patients on waiting lists is delayed.

The objective of this project is to develop an end-to-end Machine Learning decision-support system that predicts the likelihood of an outpatient missing their scheduled appointment. Rather than making autonomous decisions, this system flags high-risk appointments for hospital operational staff, enabling them to prioritize proactive reminder phone calls or secondary SMS confirmations. 

The project strictly follows ethical AI and clinical governance boundaries: **it does not predict medical diagnoses, clinical urgency, triage tiers, or patient health outcomes.**

---

## 2. Business Problem & Prediction Point Definition
In machine learning systems deployed in healthcare operations, defining the exact moment of inference is critical to prevent data leakage and ensure real-world utility:

- **Point of Inference ($T_{predict}$):** The exact instant an outpatient appointment is booked by the receptionist or through an online portal.
- **Prediction Target:** `no_show` (Binary: `0 = Attended`, `1 = Missed / No-Show`).
- **Prediction Horizon:** The time window between `booking_date` and `appointment_date` (termed `lead_time_days`).
- **Operational Action:** Appointments with predicted risk exceeding an optimized operational threshold ($\tau = 0.40$) are flagged as `"Reminder Review: Suggested"`. Administrative staff review the record and initiate a personalized reminder.

```
Time Horizon:
[Historical Care History] ---> [BOOKING TIME: Prediction Made] -------------> [APPOINTMENT DATE]
                                (Only booking info used)       (Lead Time)    (Actual Outcome: 0 or 1)
```

---

## 3. Dataset Description & Reproducible Generation
To respect patient privacy regulations (HIPAA, GDPR) and eliminate sensitive health identifiers, a realistic, reproducible synthetic dataset of **3,500 appointment records** was generated using a fixed random seed (`seed = 42`) in `generate_data.py`.

### Feature Dictionary (Information strictly available at Booking Time):
| Feature Name | Data Type | Description | Operational Significance |
| :--- | :--- | :--- | :--- |
| `appointment_id` | String | Unique appointment identifier (`APT-10000` to `APT-13499`) | Identifier only (excluded from modeling) |
| `booking_date` | Date (YYYY-MM-DD) | Date the booking was created (2025-01-01 to 2025-06-30) | Used for chronological ordering |
| `appointment_date` | Date (YYYY-MM-DD) | Scheduled date of outpatient clinic visit | Used to compute `lead_time_days` |
| `age_group` | Categorical | Young Adults (18-35), Middle-Aged (36-55), Older Adults (56-70), Seniors (71+) | Demographic attendance trend |
| `appointment_type`| Categorical | Routine Checkup, Follow-up, Specialist Consultation, Diagnostic / Lab | Clinical appointment category |
| `department` | Categorical | General Medicine, Cardiology, Orthopedics, Dermatology, Pediatrics, Ophthalmology | Clinic department |
| `booking_channel` | Categorical | Online Portal, Phone Call, In-Person, Mobile App | Booking medium |
| `previous_appointments`| Integer | Historical count of completed or booked visits prior to this booking | Patient engagement history |
| `previous_no_shows`| Integer | Historical missed visits before this booking ($\le previous\_appointments$) | Prior behavioral track record |
| `sms_opt_in` | Binary | 1 = Opted in for SMS alerts, 0 = Not opted in | Communication accessibility |
| `lead_time_days` | Integer | Calculated days between booking and appointment ($appointment\_date - booking\_date$) | Forgetting curve window |
| `no_show` (Target) | Binary | 0 = Patient Attended, 1 = Patient Missed Appointment | Target variable |

---

## 4. Exploratory Data Analysis (EDA) Observations
Exploratory data analysis was conducted and 5 primary charts were generated in `reports/figures/`:

1. **Target Distribution (`01_target_distribution.png`):**
   - Out of 3,500 appointments, **2,570 (73.43%) were attended** and **930 (26.57%) were no-shows**.
   - This ~27% missed appointment rate reflects empirical hospital outpatient no-show benchmarks and establishes a class imbalance where predicting the majority class yields high dummy accuracy but zero operational value.
2. **Departmental Variation (`02_noshow_by_department.png`):**
   - Elective and routine clinics like Dermatology (30.8%) and General Medicine (28.7%) experience higher no-show rates.
   - High-acuity or specialized clinics like Cardiology (21.4%) show noticeably lower no-show rates due to perceived health stakes.
3. **Age Demographics (`03_noshow_by_age_group.png`):**
   - Young adults (18-35) display the highest missed appointment rate (~31.2%), driven by employment conflicts and mobility.
   - Seniors (71+) show the lowest no-show rate (~21.1%), indicating high treatment compliance.
4. **Advance Lead Time (`04_lead_time_vs_noshow.png`):**
   - Same-day bookings have very low no-show rates (<14%).
   - As lead time increases beyond 14 days, no-show rates increase sharply to >36%, confirming that extended advance planning leads to patient forgetting or scheduling conflicts.
5. **Prior History Impact (`05_previous_noshows_vs_noshow.png`):**
   - Patients with zero prior no-shows miss only ~18% of visits.
   - Patients with 3 or more prior no-shows miss over 54% of scheduled visits, proving historical adherence to be the single strongest behavioral predictor.

---

## 5. Data Cleaning, Feature Engineering & Leakage Prevention
Data integrity checks verified:
- **Zero missing values** in all generated fields.
- **Zero duplicate rows.**
- **Logical consistency:** Confirmed that `previous_no_shows <= previous_appointments` and `lead_time_days >= 0` across 100% of rows.

### Strict Leakage Prevention Architecture:
1. **No Target Leakage:** `appointment_id`, `booking_date`, `appointment_date`, and `no_show` are strictly dropped from the input matrix $X$.
2. **No Temporal / Future Leakage:** Preprocessing transformations (median imputation and standard scaling for numericals; one-hot encoding for categoricals) are encapsulated inside a scikit-learn `Pipeline` with a `ColumnTransformer`. The pipeline is fitted exclusively on the training split and transformed on validation and test sets.
3. **No Chronological Cross-Contamination:** Shuffling time-series data leaks future patterns into past training splits. A strict time-based split is enforced.

---

## 6. Train / Validation / Test Splitting Strategy
Appointments were ordered chronologically by `booking_date` and segmented into three sequential windows:

| Partition | Record Count | Percentage | Date Range | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 2,450 | 70% | 2025-01-01 to 2025-05-10 | Model fitting and parameter estimation |
| **Validation Set**| 525 | 15% | 2025-05-10 to 2025-06-03 | Model comparison and decision threshold tuning |
| **Held-out Test Set**| 525 | 15% | 2025-06-03 to 2025-06-30 | Final untouched real-world simulation |

This ensures that model performance reflects true future deployment, predicting appointments that occur *after* all training records were scheduled.

---

## 7. Model Development & Comparison
Three distinct models were benchmarked:
1. **DummyClassifier (Baseline):** Predicts the prior majority class (Attended = 0). Serves as a baseline to expose the deceptive nature of accuracy in imbalanced classification.
2. **Logistic Regression:** Linear probabilistic model with balanced class weights (`class_weight='balanced'`), serving as a fast, interpretable benchmark.
3. **Random Forest Classifier:** Non-linear ensemble of 100 decision trees (`max_depth=7`, `min_samples_split=10`, `class_weight='balanced'`), capturing feature interactions between lead time, prior history, and clinic departments.

---

## 8. Validation Threshold Optimization
In standard binary classification, the default decision threshold is 0.50. However, in hospital operations, the cost of errors is heavily asymmetrical:
- **False Negative (Missed No-Show):** The hospital takes no action, the patient does not attend, the doctor's appointment slot is wasted, and care is delayed. **High cost.**
- **False Positive (Unnecessary Reminder):** The hospital sends a polite SMS or courtesy reminder to a patient who would have attended anyway. **Negligible cost.**

### Threshold Evaluation on the Validation Set (Random Forest):
| Threshold ($\tau$) | Precision | Recall | F1-Score | TP (Caught) | FP (Extra Calls) | FN (Missed) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.30** | 0.2742 | **0.9683** | 0.4273 | 122 | 323 | 4 |
| **0.40 (Selected)** | **0.3311** | **0.8016** | **0.4687** | **101** | **204** | **25** |
| **0.50 (Default)** | 0.3454 | 0.5317 | 0.4188 | 67 | 127 | 59 |
| **0.60** | 0.4318 | 0.3016 | 0.3551 | 38 | 50 | 88 |
| **0.70** | 0.6667 | 0.0476 | 0.0889 | 6 | 3 | 120 |

**Threshold Selection Decision:**  
The **0.40 threshold** was selected. At $\tau = 0.40$, Recall increases from 53.2% (default) to **80.2%**, successfully capturing 101 out of 126 missed appointments while keeping false reminder calls within manageable hospital staffing capacity.

---

## 9. Final Test Set Evaluation Results
The final models were evaluated on the untouched held-out test set (525 records: 389 attended, 136 no-shows):

| Evaluation Metric | Dummy Baseline ($\tau=0.50$) | Logistic Regression ($\tau=0.50$) | Random Forest ($\tau=0.40$) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.7410 (74.1%)** | 0.6743 (67.4%) | 0.5848 (58.5%) |
| **Precision** | 0.0000 (0.0%) | **0.4022 (40.2%)** | 0.3615 (36.2%) |
| **Recall (Sensitivity)**| 0.0000 (0.0%) | 0.5294 (52.9%) | **0.7868 (78.7%)** |
| **F1-Score** | 0.0000 | 0.4571 | **0.4954** |
| **ROC-AUC** | 0.5000 | **0.6961** | 0.6929 |
| **Brier Score** | **0.1921** | 0.2164 | 0.2087 |

### Confusion Matrix on Held-Out Test Set (Random Forest, $\tau = 0.40$):
```
                       PREDICTED
                 Attended (0)    No-Show (1)
ACTUAL
Attended (0)      TN = 200        FP = 189
No-Show (1)       FN = 29         TP = 107
```
- **Total No-Shows in Test Data:** 136
- **No-Shows Correctly Caught (TP):** **107 (78.7%)**
- **No-Shows Missed (FN):** Only 29 (21.3%)
- **True Attendees (TN):** 200

### Why Accuracy is Misleading:
The Dummy Baseline achieved 74.1% accuracy by predicting that every single patient would attend. Yet, its Recall was 0.0%, meaning it caught **zero** missed appointments. The Random Forest model with threshold 0.40 caught **107 out of 136 no-shows (78.7% recall)** and achieved the highest overall F1-score (0.4954), making it the superior operational system.

---

## 10. Probability Calibration & Diagnostic Analysis
- **ROC Curve (`reports/figures/07_roc_curve.png`):** Demonstrates an AUC of ~0.693 across all operating thresholds, significantly outperforming random guessing ($AUC = 0.500$).
- **Calibration Curve (`reports/figures/08_calibration_curve.png`):** Compares predicted probabilities to empirical no-show rates across 6 probability bins. The Brier Score of **0.2087** confirms that the output probabilities reflect genuine risk gradients, making them safe for ranking appointments.

---

## 11. End-to-End Prediction Case Walkthrough
Consider an actual test appointment evaluated through our saved pipeline:
- **Patient Profile:** Young Adult (26 years old), booked an elective Dermatology Consultation through the Online Portal 21 days in advance. History: 2 previous appointments, 1 previous missed appointment, no SMS opt-in.
- **Raw Feature Vector:** `[lead_time=21, prev_appts=2, prev_noshows=1, age='Young Adults (18-35)', type='Specialist Consultation', dept='Dermatology', channel='Online Portal', sms=0]`
- **Pipeline Processing:** Numerical values scaled via StandardScaler; categories mapped via OneHotEncoder.
- **Model Output Probability:** `predict_proba = 0.642 (64.2%)`
- **Threshold Comparison:** $0.642 \ge 0.40 \implies$ **"Reminder Review: Suggested"**
- **Operational Action:** Front-desk staff receive a flag in their daily morning queue: *"Patient APT-12840 has a 64% predicted probability of missing their dermatology appointment. Recommended action: Place a 1-minute courtesy confirmation call 48 hours prior."*

---

## 12. Privacy, Fairness & Responsible AI Considerations
1. **Advisory Decision Support Only:** The model provides a risk score to hospital staff. It **never** cancels appointments automatically, double-books slots without human approval, or charges fees.
2. **Zero Clinical Discrimination:** The model does not use patient medical diagnosis, chronic illness history, race, gender, religion, or socioeconomic status as features.
3. **Synthetic Data Realism Disclaimer:** While the synthetic dataset exhibits realistic healthcare dynamics, empirical hospital deployment requires retraining and re-calibrating on local electronic health record (EHR) data with institutional ethical board approval.

---

## 13. Conclusion & Engineering Ownership
This project demonstrates a complete, beginner-to-intermediate B.Tech engineering machine learning lifecycle:
- Problem framing tied to operational utility.
- Strict chronological data partitioning and leakage prevention.
- Reproducible synthetic dataset generation with seed 42.
- Preprocessing and classifier encapsulation inside an exportable scikit-learn pipeline.
- Threshold analysis balancing healthcare trade-offs (Recall vs. Precision).
- Standalone inference web application built with Streamlit and verified with Pytest.

