# 🎓 Faculty & Interview Viva Guide: Hospital Appointment No-Show Prediction
**Assessment Body:** ALGIONS AI LABS PRIVATE LIMITED  
**Role:** AI & Machine Learning Intern  
**Project:** Hospital Appointment No-Show Prediction  
**Level:** Final-Year B.Tech Computer Science (Data Science / ML)  

---

## 🕒 3-to-5 Minute Demonstration & Presentation Script

Use this exact script when presenting to your faculty or internship interviewer:

> *"Good morning/afternoon, Respected Faculty and Interviewers.*  
> *My name is [Your Name], and I am a final-year B.Tech Computer Science student specializing in Data Science and Machine Learning. Today, I am presenting my project for the ALGIONS AI LABS internship assessment: **Hospital Appointment No-Show Prediction**.*
>
> ***1. The Problem Statement:***  
> *In hospital outpatient clinics, patient no-shows cause idle doctor time, delayed care for waiting patients, and lost revenue. However, receptionists cannot manually call every single scheduled patient. Our goal is to build an administrative decision-support machine learning model that predicts which appointments have a high probability of being missed, so staff can send targeted reminder calls.*
>
> ***2. Ethical & Clinical Boundaries:***  
> *I want to emphasize a core requirement: this system **does NOT** predict medical diagnosis, clinical urgency, triage tier, or health outcomes. It is strictly an operational tool, and all suggested reminders require human staff review.*
>
> ***3. Data & Leakage Prevention:***  
> *I generated a reproducible synthetic dataset of 3,500 appointments with seed 42. Crucially, the prediction time is the exact moment of booking. I used only information known at booking time: lead time, patient visit history, department, booking channel, and age group. To prevent temporal data leakage, I implemented a **chronological split** (oldest 70% for training, next 15% for validation, latest 15% for testing), rather than random shuffling.*
>
> ***4. Preprocessing & Models:***  
> *All preprocessing—median imputation, standard scaling, and one-hot encoding—is encapsulated inside a Scikit-Learn `ColumnTransformer` and `Pipeline`. I trained and compared three models:*
> *1. A Dummy Baseline (which achieved 74.1% accuracy but 0% recall, proving accuracy is misleading on imbalanced data).*
> *2. Logistic Regression with balanced class weights.*
> *3. Random Forest Classifier.*
>
> ***5. Decision Threshold & Results:***  
> *Because a missed patient is far more costly to a hospital than a courtesy reminder call, I conducted a threshold analysis on validation data and selected **threshold = 0.40**. On the untouched test set, Random Forest achieved **78.7% Recall (catching 107 of 136 missed visits)**, an ROC-AUC of **0.693**, and an F1-score of **0.495**.*
>
> ***6. Deployment & Testing:***  
> *The entire fitted pipeline and threshold are serialized with Joblib and deployed in an interactive Streamlit web application. The codebase is tested with 8 passing Pytest unit tests.*
>
> *Now, let me demonstrate the Streamlit application and show a live prediction... Thank you, and I am ready for your questions."*

---

## 🛠️ Complete Project Workflow (Hinglish + English Summary)

```
[generate_data.py] ──> [appointments.csv] ──> [run_eda.py] ──> [eda.ipynb & 5 Charts]
         │
         ▼
  [train_model.py]
  • Chronological Split (70% Train, 15% Val, 15% Test)
  • ColumnTransformer (Imputer + Scaler + OneHot)
  • Models: Dummy, Logistic Regression, Random Forest
  • Validation Threshold Optimization (tau = 0.40)
  • Save Artifact: best_model.joblib & model_results.csv
         │
         ├──────────────────────────────┐
         ▼                              ▼
  [evaluate_model.py]            [app.py (Streamlit)]
  • Confusion Matrix (06)        • Section A: Home & Disclaimer
  • ROC Curve (07)               • Section B: Prediction Form
  • Calibration Curve (08)       • Section C: Benchmark Report
         │                              │
         └──────────────┬───────────────┘
                        ▼
               [tests/test_pipeline.py]
               (8 Pytest Unit Tests)
```

### Hinglish Concept Breakdown:
- **Project ka core idea kya hai?** Hospital me jab patient appointment book karta hai, hum usi waqt (`booking time`) par predict karte hain ki kya patient ke aane ke chances kam hain? Agar probability 40% se zyada aati hai, to staff unhe ek phone call ya reminder SMS bhejta hai.
- **Why at booking time?** Agar hum appointment ke baad ka data use karenge (jaise SMS deliver hua ya nahi), to use **data leakage** kehte hain. Booking time par jo info available hoti hai, sirf wahi inputs me di gayi hai.
- **Why chronological split?** Real life me past ke data se model train hota hai aur future appointments predict karni hoti hain. Agar random train-test split karenge, to future ka pattern past me leak ho jayega. Isliye chronological order follow kiya.

---

## 📚 30 Technical Viva Questions & Answers

### Category 1: Problem Definition & Data Science Basics

#### Q1. What is the business problem your project solves?
**Answer:** It addresses hospital operational inefficiencies caused by patient no-shows. By predicting attendance probability at the time of booking, hospital administrators can allocate reminder phone calls and optimize schedule buffers without wasting staff hours calling everyone.

#### Q2. Why must this project NOT predict medical diagnosis or urgency?
**Answer:** Because predicting diagnosis or urgency without full diagnostic equipment and clinical expertise introduces severe medical malpractice risks. Our model is purely operational and administrative.

#### Q3. What is the target variable and its distribution?
**Answer:** The target is `no_show` (`0 = Attended`, `1 = Missed`). In our 3,500 record dataset, 73.4% attended and 26.6% missed, representing realistic class imbalance.

#### Q4. What is Data Leakage, and how did you prevent it?
**Answer (English):** Data leakage happens when information from outside the training dataset (or future information) is inadvertently used to create the model, leading to overly optimistic results that fail in production.  
**Hinglish:** Jab model training ke dauran aisi information use kar leta hai jo real prediction ke time available nahi hogi (future data ya test data), use data leakage kehte hain.  
**Prevention in this project:**
1. Dropped `appointment_id`, dates, and target from features.
2. Put all imputers and encoders inside a `Pipeline` fitted *only* on training data.
3. Used a chronological split so future appointments are never seen during training.

#### Q5. Why is chronological splitting preferred over random `train_test_split`?
**Answer:** Time-dependent data has temporal trends. Random splitting leaks future dates into training data. A chronological split (Oldest 70% Train, Next 15% Val, Latest 15% Test) mirrors real hospital deployment where models are trained on past data and evaluated on future appointments.

---

### Category 2: Feature Engineering & Preprocessing

#### Q6. What features did you use, and why?
**Answer:** We used features known at booking time:
- `lead_time_days` ($appointment\_date - booking\_date$): Longer lead times increase forgetting.
- `previous_appointments` & `previous_no_shows`: Strongest historical predictor of attendance behavior.
- `department`: Certain clinics (e.g. Dermatology) have higher cancellation rates than critical ones (Cardiology).
- `age_group`: Captures demographic attendance tendencies.
- `booking_channel` & `sms_opt_in`: Signals patient engagement and contactability.

#### Q7. What is `ColumnTransformer` and why did you use it?
**Answer (English):** `ColumnTransformer` applies different transformations to different subsets of features. We transformed numerical features using `SimpleImputer` and `StandardScaler`, while categorical features were processed with `SimpleImputer` and `OneHotEncoder`.  
**Hinglish:** Numerical aur categorical columns ko alag-alag handle karne ke liye Scikit-Learn ka `ColumnTransformer` use kiya jata hai. Ek hi step me dono process ho jate hain.

#### Q8. What is the difference between `fit()`, `transform()`, and `fit_transform()`?
**Answer:**
- `fit()`: Learns parameters from data (e.g., mean and standard deviation in `StandardScaler`, categories in `OneHotEncoder`).
- `transform()`: Applies learned parameters to transform raw data into scaled/encoded features.
- `fit_transform()`: Does both in one step on training data. Never call `fit_transform()` on validation/test data!

#### Q9. What does `Pipeline` do?
**Answer:** A `Pipeline` chains together preprocessing steps and an estimator into a single object. Calling `pipeline.fit(X_train, y_train)` fits the preprocessor and then fits the model. Calling `pipeline.predict(X_test)` automatically preprocesses the test data using training parameters and returns predictions, eliminating manual leakage.

#### Q10. Why did you use `SimpleImputer` with median and most-frequent?
**Answer:** To make the production pipeline resilient to missing values in real-time Streamlit inputs. Median is robust to numerical outliers, and most-frequent fills categorical gaps with mode.

---

### Category 3: Machine Learning Algorithms

#### Q11. Why did you include a `DummyClassifier`?
**Answer:** It acts as a baseline benchmark. Since 73.4% of patients attend, a dummy classifier predicting the majority class achieves 74.1% accuracy while achieving 0% recall. It proves to reviewers that high accuracy can be completely useless in imbalanced problems.

#### Q12. Why did you choose `LogisticRegression`?
**Answer:** Logistic Regression is a linear statistical model that estimates log-odds of the positive class using the sigmoid function:
$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$
It provides a transparent, well-calibrated probabilistic baseline with minimal computation.

#### Q13. Why did you choose `RandomForestClassifier`?
**Answer:** Random Forest is an ensemble of decision trees using bagging (Bootstrap Aggregation) and feature subsampling. It captures non-linear relationships (e.g. high lead time combined with prior no-shows) and feature interactions without severe overfitting.

#### Q14. What does `class_weight='balanced'` do?
**Answer (English):** It adjusts weights inversely proportional to class frequencies:
$$w_j = \frac{N}{2 \times N_j}$$
This penalizes misclassifications of the minority class (no-shows) more heavily during loss calculation, preventing the model from ignoring the minority class.  
**Hinglish:** Minority class (jo kam baar aati hai) ke misclassification par zyada penalty lagata hai taaki model sirf majority class predict na kare.

#### Q15. How did Random Forest prevent overfitting?
**Answer:** By constraining tree depth (`max_depth=7`) and minimum sample split (`min_samples_split=10`), and training across 100 trees with random feature subsampling.

---

### Category 4: Evaluation Metrics & Healthcare Trade-offs

#### Q16. Define True Positive, False Positive, True Negative, False Negative in this hospital context.
**Answer:**
- **True Positive (TP):** Model predicts no-show; patient actually missed appointment. (Successful reminder opportunity!)
- **False Positive (FP):** Model predicts no-show; patient actually attends. (Minor cost: extra reminder call).
- **True Negative (TN):** Model predicts attended; patient attended. (No action needed).
- **False Negative (FN):** Model predicts attended; patient missed appointment. (High cost: wasted doctor slot!)

#### Q17. What is Precision and Recall, and which one is more important here?
**Answer:**
- $\text{Precision} = \frac{TP}{TP + FP}$: Out of all appointments flagged for reminder, how many actually would miss?
- $\text{Recall} = \frac{TP}{TP + FN}$: Out of all appointments that missed, how many did we catch?
- **Which is more important?** **Recall is far more important.** In hospital management, missing a no-show (FN) wastes 45 minutes of clinical time. A false alarm (FP) only costs a 30-second reminder SMS.

#### Q18. What is the F1-Score?
**Answer:** The harmonic mean of Precision and Recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It balances both metrics into a single score. Random Forest achieved an F1 of 0.4954.

#### Q19. What is ROC-AUC?
**Answer:** Receiver Operating Characteristic - Area Under Curve measures the classifier's ability to discriminate between attendees and no-shows across all possible thresholds. An AUC of 0.5 is random guessing; our models achieved ~0.693 to 0.696.

#### Q20. What is a Brier Score, and what does it measure?
**Answer:** The Brier score measures the mean squared error between predicted probabilities and actual binary outcomes ($0$ or $1$):
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Lower scores indicate better-calibrated probabilities. Our model achieved ~0.2087.

---

### Category 5: Decision Threshold Optimization

#### Q21. Why did you change the decision threshold from 0.50 to 0.40?
**Answer (English):** The default 0.50 threshold produced only 53.2% recall on validation data, missing 59 no-shows. By lowering the threshold to 0.40, recall increased to 80.2% on validation and 78.7% on the held-out test set, capturing 107 out of 136 missed appointments.  
**Hinglish:** 0.5 threshold par bohot saare no-shows miss ho rahe the. Hospital ke liye ek extra reminder phone call bhejna bohot sasta hai, par doctor ka time waste hona bohot mehenga hai. Isliye threshold ko 0.40 par set kiya taaki 78.7% no-shows catch ho sakein.

#### Q22. How was the threshold chosen without test set leakage?
**Answer:** The threshold analysis was performed strictly on the **Validation set** (the middle 15% chronologically). Once 0.40 was selected, it was frozen and evaluated on the untouched **Test set**.

---

### Category 6: Streamlit Web App & Software Engineering

#### Q23. How does the Streamlit application generate predictions?
**Answer:**
1. The user inputs appointment details in the UI form.
2. The app automatically calculates `lead_time_days = (appointment_date - booking_date).days`.
3. Input validation confirms dates are valid and prior no-shows $\le$ prior visits.
4. The saved pipeline is loaded via `@st.cache_resource` using `joblib.load()`.
5. `pipeline.predict_proba(input_df)[0, 1]` generates the probability.
6. The probability is compared to the saved threshold (0.40) to display `"Suggested"` or `"Not Prioritized"`.

#### Q24. Why did you use `@st.cache_resource` in Streamlit?
**Answer:** It caches the loaded model in server memory so it doesn't re-read the 5 MB joblib file from disk on every user interaction, making the web UI fast and responsive.

#### Q25. What is stored in `best_model.joblib`?
**Answer:** A Python dictionary containing:
- The fitted `Pipeline` (including `ColumnTransformer` and `RandomForestClassifier`).
- The chosen operating `threshold` (0.40).
- The `feature_names` and feature categories.
- Validation and test metric dictionaries for governance.

#### Q26. Why did you use Pytest?
**Answer:** Pytest automates verification of our pipeline. We wrote 8 unit tests checking dataset size, schema, negative date handling, artifact loading, probability bounds $[0, 1]$, and prediction consistency.

---

### Category 7: Ethics, Limitations & Governance

#### Q27. What are the key limitations of your model?
**Answer:**
1. Trained on synthetic data; real hospital deployments require retraining on electronic health records (EHR).
2. It does not account for real-time external events (e.g. heavy monsoon rain, transit strikes).
3. Predictions are probabilistic correlations, not causal guarantees.

#### Q28. How does this system ensure fairness and prevent algorithmic discrimination?
**Answer:** We deliberately omitted sensitive demographic attributes like religion, race, annual income, or medical diagnoses from the feature matrix. Furthermore, the model is strictly advisory and cannot deny appointments or auto-cancel bookings.

#### Q29. What happens if a patient is falsely flagged as a high-risk no-show?
**Answer:** They receive a polite, standard appointment confirmation call or text message from hospital staff asking if they can make their scheduled visit or wish to reschedule. No punitive action is taken.

#### Q30. If you had 2 more weeks on this internship project, what would you improve?
**Answer:**
1. Integrate automated SMS dispatch using the Twilio / WhatsApp Business API.
2. Implement patient-level clustering to segment chronic vs acute visit behaviors.
3. Incorporate local weather forecast APIs to capture rain/extreme heat signals 24 hours prior to appointments.

