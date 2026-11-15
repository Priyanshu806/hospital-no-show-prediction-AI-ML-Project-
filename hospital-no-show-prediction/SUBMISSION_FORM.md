# 📋 ALGIONS AI LABS Internship Assessment Submission Form
### Assigned Project: Hospital Appointment No-Show Prediction
**Track:** AI & Machine Learning | Advanced Individual Project Assessment  

*Fill out the placeholders marked in brackets `[ ... ]` before final submission.*

---

## 1. Candidate Identification
| Field | Details |
| :--- | :--- |
| **Student Name** | [Your Full Name] |
| **Registration / Roll Number** | [Your University Registration / Roll Number] |
| **University Email** | [Your Official University / College Email ID] |
| **Personal Email** | [Your Personal Email ID] |
| **Degree & Year** | B.Tech Computer Science & Engineering (Final Year) |
| **Specialization** | Data Science / Machine Learning |

---

## 2. Assignment Confirmation
| Field | Details |
| :--- | :--- |
| **Internship Role** | AI & Machine Learning |
| **Assigned Project Title** | Hospital Appointment No-Show Prediction |
| **Assigned Domain** | Hospital Management & Healthcare Operations |
| **Submission Date** | [Date of Submission, e.g., November 2026] |

---

## 3. Code Repository & Artifacts
| Field | Details |
| :--- | :--- |
| **Source Repository URL** | [GitHub Repository Link, e.g., https://github.com/your-username/hospital-no-show-prediction] |
| **Primary Branch** | `main` |
| **Final Commit ID** | [Your Final Git Commit Hash, e.g., run `git rev-parse HEAD`] |
| **Reviewer Access** | Public (or Private with reviewer username invited) |

---

## 4. Demonstration & Deployment
| Field | Details |
| :--- | :--- |
| **3–5 Minute Video Link** | [Loom / Google Drive / YouTube Unlisted Link] |
| **Live App URL** | Local-Only Deployment (Streamlit) / [Streamlit Community Cloud Link if deployed] |
| **Local Run Command** | `cd hospital-no-show-prediction; streamlit run app.py` |

---

## 5. Technical Documentation Links
| Field | Details |
| :--- | :--- |
| **README Link** | [`README.md`](README.md) in repository root |
| **Technical Report Link** | [`reports/technical_report.md`](reports/technical_report.md) |
| **Faculty & Viva Guide** | [`VIVA_GUIDE.md`](VIVA_GUIDE.md) |
| **Architecture Diagram** | Rendered via Mermaid in `README.md` (Section 3) |

---

## 6. Technology Stack & Environment Setup
| Category | Stack |
| :--- | :--- |
| **Programming Language** | Python 3.13 / 3.10+ |
| **Core Libraries** | Pandas (2.2.3), NumPy (2.2.2), Scikit-learn (1.6.1), Matplotlib (3.10.1), Seaborn (0.13.2) |
| **Model Serialization** | Joblib (1.5.0) |
| **Interactive Interface** | Streamlit (1.44.0) |
| **Automated Verification** | Pytest (9.2.0) — 8 / 8 tests passing |
| **Setup Notes** | `pip install -r requirements.txt` followed by `streamlit run app.py` |

---

## 7. Role-Specific Technical Fields

### A. Dataset Source & Attribution
- **Dataset Strategy:** Reproducible synthetic dataset generated via `generate_data.py`.
- **Random Seed:** `42`
- **Total Records:** 3,500 appointments (exceeds the 1,000 / 3,000 threshold requirement).
- **Target Variable:** `no_show` (0 = Attended [73.4%], 1 = Missed [26.6%]).
- **Leakage Prevention:** Features restricted strictly to information available at booking time (`lead_time_days`, `previous_appointments`, `previous_no_shows`, `department`, `age_group`, `booking_channel`, `sms_opt_in`). No post-booking or clinical outcome data.

### B. Machine Learning Modeling & Comparison
- **Baseline Model:** `DummyClassifier(strategy='prior')` $\rightarrow$ Accuracy: 74.1%, Recall: 0.0%, F1: 0.000.
- **Linear Benchmark:** `LogisticRegression(class_weight='balanced')` $\rightarrow$ Accuracy: 67.4%, Precision: 40.2%, Recall: 52.9%, F1: 0.457, ROC-AUC: 0.696.
- **Selected Champion Model:** `RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=10, class_weight='balanced')` with Scikit-learn `Pipeline` and `ColumnTransformer`.
- **Threshold Tuning:** Selected $\tau = 0.40$ on chronological validation data to maximize recall for patient outreach.

### C. Test Set Evaluation Metrics (Held-Out Future Data, N=525)
- **Accuracy:** 58.5%
- **Precision:** 36.2%
- **Recall (Sensitivity):** **78.7%** (Caught 107 out of 136 missed appointments)
- **F1-Score:** **0.4954**
- **ROC-AUC:** **0.6929**
- **Brier Score:** **0.2087**
- **Confusion Matrix:** True Negative (TN) = 200, False Positive (FP) = 189, False Negative (FN) = 29, True Positive (TP) = 107

### D. Model Artifact Location
- Saved in repository at: `models/best_model.joblib` (contains fitted Pipeline, threshold, feature metadata, and benchmark results).

---

## 8. AI & Attribution Declarations
- **AI Tools Used:** Antigravity (Google DeepMind)
- **Purpose of Use:** Architecture structuring, code templating, test harness configuration, and formatting documentation.
- **Independent Verification:** Candidate independently understands every function, pipeline stage, and metric, and can debug or modify the codebase live during the technical interview.

---

## 9. Compliance Declarations
- [x] **Originality:** The code, analysis, and architecture represent original engineering effort adhering to assessment criteria.
- [x] **No Restricted / Sensitive Data:** No real patient names, contact numbers, diagnoses, or clinical secrets are stored.
- [x] **Reproducibility:** The entire pipeline can be rerun from scratch with documented commands.
- [x] **Working Interface:** The Streamlit application performs real inference from the serialized Joblib pipeline without mock or hard-coded results.
- [x] **Human-in-the-Loop:** Explicitly disclaims medical diagnosis and establishes predictions as advisory decision support for human review.

