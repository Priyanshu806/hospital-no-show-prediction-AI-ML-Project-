"""
evaluate_model.py - Standalone Model Evaluation & Diagnostic Visualizer
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This script loads the saved trained pipeline artifact from models/best_model.joblib,
runs full diagnostics on the held-out chronological test set, and generates:
1. Confusion Matrix heatmap (reports/figures/06_confusion_matrix.png)
2. ROC Curve comparison (reports/figures/07_roc_curve.png)
3. Probability Calibration Curve / Reliability Diagram (reports/figures/08_calibration_curve.png)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def evaluate_and_plot():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "best_model.joblib")
    data_path = os.path.join(base_dir, "data", "appointments.csv")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=====================================================================")
    print("Standalone Model Evaluation & Diagnostic Report")
    print("=====================================================================")

    # 1. Load Artifact and Data
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run train_model.py first.")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    model_name = artifact["model_name"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(data_path)
    df_sorted = df.sort_values(by="booking_date").reset_index(drop=True)
    val_end = int(0.85 * len(df_sorted))
    test_df = df_sorted.iloc[val_end:].copy()

    X_test = test_df[feature_names]
    y_test = test_df["no_show"]

    print(f"[*] Loaded Model Artifact: {model_name} (Decision Threshold: {threshold})")
    print(f"[*] Held-out Test Set Records: {len(test_df)}")

    # 2. Compute Predictions
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    # 3. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(7, 6))
    annot = [
        [f"True Negative (TN)\n{tn}\n({tn/len(y_test)*100:.1f}%)", f"False Positive (FP)\n{fp}\n({fp/len(y_test)*100:.1f}%)"],
        [f"False Negative (FN)\n{fn}\n({fn/len(y_test)*100:.1f}%)", f"True Positive (TP)\n{tp}\n({tp/len(y_test)*100:.1f}%)"]
    ]
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Attended (0)", "Predicted No-Show (1)"],
        yticklabels=["Actual Attended (0)", "Actual No-Show (1)"],
        linewidths=1.5,
        linecolor="black"
    )
    plt.title(f"Confusion Matrix: {model_name} (Threshold = {threshold})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model Decision (Reminder Recommendation)", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Patient Attendance", fontsize=12, labelpad=10)
    cm_path = os.path.join(fig_dir, "06_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix: {cm_path}")

    # 4. Generate ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    
    # Mark the chosen operating point
    operating_fpr = fp / (fp + tn)
    operating_tpr = tp / (tp + fn)
    plt.scatter([operating_fpr], [operating_tpr], color="#e74c3c", s=100, zorder=5, 
                label=f"Selected Operating Point (t={threshold})\n(Recall={operating_tpr:.2f}, FPR={operating_fpr:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Unnecessary Reminders / All Attendees)", fontsize=11)
    plt.ylabel("True Positive Rate / Recall (Caught No-Shows / All Missed)", fontsize=11)
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    roc_path = os.path.join(fig_dir, "07_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC Curve: {roc_path}")

    # 5. Generate Probability Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=6, strategy="uniform")
    brier = brier_score_loss(y_test, y_test_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2.5, color="#8e44ad", label=f"{model_name} (Brier Score = {brier:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability of No-Show", fontsize=11)
    plt.ylabel("Observed Empirical Proportion of No-Shows", fontsize=11)
    plt.title("Probability Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    calib_path = os.path.join(fig_dir, "08_calibration_curve.png")
    plt.savefig(calib_path, dpi=300)
    plt.close()
    print(f"[+] Saved Calibration Curve: {calib_path}")

    # Print Summary Table
    print("\n================ FINAL TEST PERFORMANCE SUMMARY ================")
    print(f"Accuracy         : {artifact['test_metrics']['Accuracy']:.4f}")
    print(f"Precision        : {artifact['test_metrics']['Precision']:.4f}")
    print(f"Recall (Sens.)   : {artifact['test_metrics']['Recall']:.4f}  <-- Caught {tp} of {tp + fn} no-shows")
    print(f"F1-Score         : {artifact['test_metrics']['F1-Score']:.4f}")
    print(f"ROC-AUC          : {artifact['test_metrics']['ROC-AUC']:.4f}")
    print(f"Brier Score      : {artifact['test_metrics']['Brier Score']:.4f}")
    print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=================================================================\n")


if __name__ == "__main__":
    evaluate_and_plot()

