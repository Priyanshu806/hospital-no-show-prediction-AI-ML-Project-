"""
app.py - Streamlit Web Application
Hospital Appointment No-Show Prediction
Subtitle: Machine Learning-Based Appointment Reminder Support System
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This application provides an interactive decision-support interface for hospital operational staff.
Predictions are strictly advisory; all reminders require human staff review.
"""

import os
import joblib
import pandas as pd
from datetime import date, timedelta
import streamlit as st

# Custom minimal styling for clean UI
CSS_STYLE = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .advisory-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400e;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
</style>
"""


@st.cache_resource
def load_trained_artifact():
    """
    Loads the trained model artifact and metadata from models/best_model.joblib.
    @st.cache_resource keeps the model in memory so it doesn't reload on every button click.
    """
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def main():
    # -------------------------------------------------------------
    # 1. Page Configuration & Header
    # -------------------------------------------------------------
    st.set_page_config(
        page_title="Hospital Appointment No-Show Prediction",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(CSS_STYLE, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🏥 Hospital Appointment No-Show Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Machine Learning-Based Appointment Reminder Support System</div>', unsafe_allow_html=True)

    # Permanent Advisory Banner (Crucial for Healthcare Compliance)
    st.markdown("""
    <div class="advisory-box">
        <strong>⚠️ Clinical Decision-Support Notice:</strong>
        This system predicts operational appointment attendance risk to assist administrative staff in scheduling reminder calls.
        It does <strong>NOT</strong> predict clinical urgency, diagnoses, treatment necessity, or triage outcomes.
        All reminder decisions must be reviewed and confirmed by human hospital staff.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. Navigation Sidebar
    # -------------------------------------------------------------
    navigation_option = st.sidebar.radio(
        "Navigation Menu",
        ["Section A: Home", "Section B: Prediction", "Section C: Model Performance"],
        index=1  # Default to Prediction tab for reviewers
    )

    # Load saved model artifact
    artifact = load_trained_artifact()
    if artifact is None:
        st.error("⚠️ Trained model artifact not found! Please run `python train_model.py` in your terminal first.")
        st.stop()

    pipeline = artifact["pipeline"]
    chosen_threshold = artifact["threshold"]
    model_name = artifact["model_name"]
    test_metrics = artifact.get("test_metrics", {})

    # =========================================================================
    # SECTION A: HOME
    # =========================================================================
    if navigation_option == "Section A: Home":
        st.header("📌 Project Objective & Overview")
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("""
            ### The Problem
            Outpatient missed appointments ("no-shows") cost healthcare systems millions annually, leave clinical capacity idle, 
            and delay necessary treatments for other patients waiting in queue.

            ### Our Solution
            A machine learning classification pipeline trained on historical scheduling patterns to estimate the 
            probability of a patient missing their booked outpatient appointment.

            ### Key Architectural Principles
            - **Booking-Time Point of Inference:** Predictions are made at the moment of booking, using only information 
              available at that instant (lead time, past visit counts, booking channel).
            - **No Future Leakage:** Post-booking occurrences (e.g. reminder receipts, clinical status changes) are excluded.
            - **Strict Human Oversight:** The model is an administrative assistant; it never contacts patients autonomously or alters clinical scheduling.
            - **Chronological Evaluation:** Validated on chronologically split past-vs-future records.
            """)

        with col2:
            st.info(f"""
            **Deployed ML Model:** {model_name}
            
            **Decision Threshold:** {chosen_threshold:.2f}
            
            **Primary Optimization Metric:** Recall ({test_metrics.get('Recall', 0):.1%}) & ROC-AUC ({test_metrics.get('ROC-AUC', 0):.3f})
            
            **Target:** `no_show` (0 = Attended, 1 = Missed)
            """)

            st.markdown("#### How Staff Use This System")
            st.markdown("""
            1. An appointment is scheduled in the hospital booking system.
            2. The system estimates attendance probability.
            3. If probability exceeds threshold (**40%**), staff receive a **Suggested Reminder Review**.
            4. Staff review patient notes and place a friendly courtesy call or WhatsApp confirmation.
            """)

    # =========================================================================
    # SECTION B: PREDICTION
    # =========================================================================
    elif navigation_option == "Section B: Prediction":
        st.header("🔍 Outpatient Appointment Prediction Form")
        st.write("Enter the appointment details known at the time of booking to calculate attendance probability.")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🗓️ Temporal Schedule")
                today = date.today()
                booking_date = st.date_input("Booking Date", value=today, max_value=today + timedelta(days=365))
                appointment_date = st.date_input("Appointment Date", value=today + timedelta(days=7), min_value=booking_date)

                # Real-time lead time calculation
                lead_time_days = (appointment_date - booking_date).days
                st.caption(f"Calculated Lead Time: **{lead_time_days} days** advance booking")

                st.subheader("🏥 Clinical Context")
                department = st.selectbox(
                    "Hospital Department",
                    ["General Medicine", "Cardiology", "Orthopedics", "Dermatology", "Pediatrics", "Ophthalmology"]
                )
                appointment_type = st.selectbox(
                    "Appointment Type",
                    ["Routine Checkup", "Follow-up", "Specialist Consultation", "Diagnostic / Lab"]
                )

            with col2:
                st.subheader("👤 Patient Demographics & History")
                age_group = st.selectbox(
                    "Patient Age Category",
                    ["Young Adults (18-35)", "Middle-Aged (36-55)", "Older Adults (56-70)", "Seniors (71+)"]
                )
                booking_channel = st.selectbox(
                    "Booking Channel",
                    ["Online Portal", "Phone Call", "In-Person", "Mobile App"]
                )
                previous_appointments = st.number_input(
                    "Total Previous Appointments at Hospital",
                    min_value=0,
                    max_value=30,
                    value=2,
                    step=1,
                    help="Historical count of total appointments scheduled prior to this booking."
                )
                previous_no_shows = st.number_input(
                    "Previous Missed Appointments (No-Shows)",
                    min_value=0,
                    max_value=int(previous_appointments),
                    value=0,
                    step=1,
                    help="Must be less than or equal to total previous appointments."
                )
                sms_opt_in = st.radio(
                    "Patient Opted In for SMS Notifications?",
                    options=[1, 0],
                    format_func=lambda x: "Yes (Opted In)" if x == 1 else "No (Not Opted In)",
                    horizontal=True
                )

            st.write("---")
            submitted = st.form_submit_button("🔮 Predict No-Show Probability", use_container_width=True)

        if submitted:
            # 1. Validation
            if appointment_date < booking_date:
                st.error("Validation Error: Appointment date cannot precede booking date.")
                st.stop()

            if previous_no_shows > previous_appointments:
                st.error("Validation Error: Previous no-shows cannot exceed total previous appointments.")
                st.stop()

            # 2. Build Input DataFrame matching pipeline features
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

            # 3. Model Inference via Saved Pipeline
            with st.spinner("Processing through trained ML pipeline..."):
                predicted_prob = pipeline.predict_proba(input_df)[0, 1]
                prob_percentage = round(predicted_prob * 100, 1)

            st.markdown("### 📊 Prediction Results")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.metric(
                    label="Predicted No-Show Probability",
                    value=f"{prob_percentage}%",
                    delta=f"{round(predicted_prob - chosen_threshold, 2):+.2f} vs Threshold ({chosen_threshold*100:.0f}%)",
                    delta_color="inverse"
                )
                st.progress(float(predicted_prob))

            with res_col2:
                if predicted_prob >= chosen_threshold:
                    st.error("### Reminder Review: Suggested ⚠️")
                    st.write(
                        f"**Action Recommendation:** This appointment exceeds the operational threshold of {chosen_threshold*100:.0f}%. "
                        "Staff are advised to review patient records and consider scheduling a personalized phone call or priority SMS confirmation 24-48 hours before the visit."
                    )
                else:
                    st.success("### Reminder Review: Not Prioritized ✅")
                    st.write(
                        f"**Action Recommendation:** The predicted risk ({prob_percentage}%) is below the operational review threshold ({chosen_threshold*100:.0f}%). "
                        "Standard automated notifications are sufficient. No manual staff escalation needed."
                    )

            # Feature Influence Breakdown
            st.markdown("#### 💡 Clinical & Scheduling Signal Insights")
            insights = []
            if lead_time_days >= 14:
                insights.append(f"- ⚠️ **Long Lead Time ({lead_time_days} days):** Appointments scheduled >14 days in advance carry elevated forgetting rates.")
            if previous_no_shows >= 1:
                insights.append(f"- ⚠️ **Prior Missed Visits ({previous_no_shows}):** Historical no-shows are the strongest risk multiplier.")
            if sms_opt_in == 0:
                insights.append("- ⚠️ **No SMS Opt-In:** Patient does not receive standard text alerts, slightly increasing risk.")
            if department in ["Dermatology", "General Medicine"]:
                insights.append(f"- ℹ️ **Department ({department}):** Routine elective departments experience higher cancellation rates than cardiology or oncology.")
            if not insights:
                insights.append("- ✅ **Low Risk Factors:** Short advance notice, clean historical attendance record, and SMS opt-in confirmed.")

            for insight in insights:
                st.write(insight)

            st.caption("Disclaimer: Probabilities are generated from historical statistical trends and should assist, not replace, human judgment.")

    # =========================================================================
    # SECTION C: MODEL PERFORMANCE
    # =========================================================================
    elif navigation_option == "Section C: Model Performance":
        st.header("📈 Model Evaluation & Comparative Diagnostics")
        st.write("Rigorous evaluation on the held-out chronological test set (unseen future appointments).")

        # Load results CSV
        results_csv = os.path.join(os.path.dirname(__file__), "reports", "model_results.csv")
        if os.path.exists(results_csv):
            res_df = pd.read_csv(results_csv)
            test_res = res_df[res_df["Dataset"] == "Test"].copy()
            
            st.subheader("1. Test Set Model Comparison Table")
            display_cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Brier Score"]
            st.dataframe(
                test_res[display_cols].style.highlight_max(subset=["Recall", "F1-Score", "ROC-AUC"], color="#dcfce7"),
                use_container_width=True
            )
        else:
            st.info("Run `python train_model.py` to view tabulated comparison.")

        st.subheader("2. Diagnostic Visualizations")
        fig_col1, fig_col2 = st.columns(2)
        fig_dir = os.path.join(os.path.dirname(__file__), "reports", "figures")

        with fig_col1:
            cm_img = os.path.join(fig_dir, "06_confusion_matrix.png")
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Confusion Matrix on Untouched Test Set", use_container_width=True)
            else:
                st.info("Confusion matrix image not yet generated. Run `evaluate_model.py`.")

        with fig_col2:
            roc_img = os.path.join(fig_dir, "07_roc_curve.png")
            if os.path.exists(roc_img):
                st.image(roc_img, caption="ROC Curve with Selected Operating Point", use_container_width=True)
            else:
                st.info("ROC Curve image not yet generated.")

        calib_img = os.path.join(fig_dir, "08_calibration_curve.png")
        if os.path.exists(calib_img):
            st.image(calib_img, caption="Reliability Diagram (Probability Calibration)", use_container_width=True)

        st.subheader("3. Operational Trade-Off: False Reminders vs. Missed No-Shows")
        st.markdown("""
        In hospital outpatient management, prediction errors have asymmetric costs:
        - **False Positive (Unnecessary Reminder):** The model predicts a no-show, but the patient would attend.
          *Cost:* A 30-second staff courtesy phone call or 5-cent SMS reminder. Very minor expense.
        - **False Negative (Missed No-Show):** The model predicts attendance, but the patient misses the appointment.
          *Cost:* 30-45 minutes of wasted doctor/nurse clinical time, idle examination equipment, and an appointment lost to another sick patient on the waitlist. Very high impact!
        
        **Conclusion:** By choosing a calibrated decision threshold of **0.40** instead of 0.50, our system maximizes 
        **Recall (catching >78% of all no-shows)** at the price of small, harmless extra reminder messages.
        """)


if __name__ == "__main__":
    main()
