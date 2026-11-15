"""
evaluate_model.py - Standalone Model Evaluation & Diagnostic Visualizer
Project: Hospital Appointment No-Show Prediction
Assessment: ALGIONS AI LABS PRIVATE LIMITED

This script loads the saved trained pipeline from models/best_model.joblib,
runs diagnostics on the held-out chronological test set, and generates:
1. Confusion Matrix heatmap (reports/figures/06_confusion_matrix.png)
2. ROC Curve comparison (reports/figures/07_roc_curve.png)
3. Probability Calibration Curve / Reliability Diagram (reports/figures/08_calibration_curve.png)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def find_file(relative_subpath):
    """Finds path regardless of current execution directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, relative_subpath),
        os.path.join(current_dir, "hospital-no-show-prediction", relative_subpath),
        os.path.join("hospital-no-show-prediction", relative_subpath),
        os.path.join(relative_subpath)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def evaluate_and_plot():
    """Loads the champion model artifact and generates test diagnostic plots."""
    model_path = find_file(os.path.join("models", "best_model.joblib"))
    data_path = find_file(os.path.join("data", "appointments.csv"))
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("==================================================")
    print("Standalone Model Evaluation & Diagnostic Report")
    print("==================================================")

    # 1. Load Model Artifact
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Artifact not found at {model_path}. Please run train_model.py first.")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    model_name = artifact["model_name"]
    feature_names = artifact["feature_names"]

    print(f"Loaded Champion Model  : {model_name}")
    print(f"Operating Threshold    : {threshold}")

    # 2. Load Held-Out Test Data
    df = pd.read_csv(data_path)
    sorted_df = df.sort_values(by="booking_date").reset_index(drop=True)
    val_end = int(0.85 * len(sorted_df))
    test_df = sorted_df.iloc[val_end:]

    X_test = test_df[feature_names]
    y_test = test_df["no_show"]
    total_test_records = len(test_df)

    print(f"Held-Out Test Records  : {total_test_records}")
    print(f"Actual No-Shows (1)    : {(y_test == 1).sum()}")
    print(f"Actual Attended (0)    : {(y_test == 0).sum()}")

    # 3. Model Predictions
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    # 4. Confusion Matrix Plot
    cm = confusion_matrix(y_test, test_predictions)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(7, 6))
    matrix_labels = [
        [f"True Negative (TN)\n{tn}\n({tn/total_test_records*100:.1f}%)",
         f"False Positive (FP)\n{fp}\n({fp/total_test_records*100:.1f}%)"],
        [f"False Negative (FN)\n{fn}\n({fn/total_test_records*100:.1f}%)",
         f"True Positive (TP)\n{tp}\n({tp/total_test_records*100:.1f}%)"]
    ]
    sns.heatmap(
        cm,
        annot=matrix_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Attended (0)", "Predicted No-Show (1)"],
        yticklabels=["Actual Attended (0)", "Actual No-Show (1)"],
        linewidths=1.5,
        linecolor="black"
    )
    plt.title(f"Confusion Matrix: {model_name} (Threshold = {threshold})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model Recommendation", fontsize=12, labelpad=10)
    plt.ylabel("Actual Patient Attendance", fontsize=12, labelpad=10)
    
    cm_path = os.path.join(figures_dir, "06_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix : {cm_path}")

    # 5. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, test_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Guessing (AUC = 0.500)")
    
    operating_fpr = fp / (fp + tn)
    operating_tpr = tp / (tp + fn)
    plt.scatter([operating_fpr], [operating_tpr], color="#e74c3c", s=100, zorder=5,
                label=f"Selected Threshold ({threshold})\n(Recall={operating_tpr:.2f}, FPR={operating_fpr:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Unnecessary Reminder Calls)", fontsize=11)
    plt.ylabel("True Positive Rate / Recall (Caught No-Shows)", fontsize=11)
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    roc_path = os.path.join(figures_dir, "07_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve        : {roc_path}")

    # 6. Probability Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, test_probabilities, n_bins=6, strategy="uniform"
    )
    brier = brier_score_loss(y_test, test_probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker="o", linewidth=2.5,
             color="#8e44ad", label=f"{model_name} (Brier Score = {brier:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability of No-Show", fontsize=11)
    plt.ylabel("Observed Proportion of No-Shows", fontsize=11)
    plt.title("Probability Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    calib_path = os.path.join(figures_dir, "08_calibration_curve.png")
    plt.savefig(calib_path, dpi=300)
    plt.close()
    print(f"Saved Calibration Curve: {calib_path}")

    # 7. Print Human-Readable Summary
    acc = artifact['test_metrics']['Accuracy']
    prec = artifact['test_metrics']['Precision']
    rec = artifact['test_metrics']['Recall']
    f1 = artifact['test_metrics']['F1-Score']
    auc_val = artifact['test_metrics']['ROC-AUC']
    brier_val = artifact['test_metrics']['Brier Score']

    print("\n================ FINAL TEST PERFORMANCE SUMMARY ================")
    print(f"Accuracy         : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision        : {prec:.4f} ({prec*100:.1f}%)")
    print(f"Recall (Sens.)   : {rec:.4f} ({rec*100:.1f}% -> Caught {tp} of {tp + fn} no-shows)")
    print(f"F1-Score         : {f1:.4f}")
    print(f"ROC-AUC          : {auc_val:.4f}")
    print(f"Brier Score      : {brier_val:.4f}")
    print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=================================================================\n")


if __name__ == "__main__":
    evaluate_and_plot()
