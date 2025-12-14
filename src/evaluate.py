# src/evaluate.py

import matplotlib
matplotlib.use("Agg")  # Backend fix (PyCharm-safe)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

OUTPUT_DIR = "models/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1) CONFUSION MATRIX
# --------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Billing", "Billing"],
        yticklabels=["No Billing", "Billing"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()

    file_path = f"{OUTPUT_DIR}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"Confusion Matrix gespeichert unter: {file_path}")


# --------------------------------------------------
# 2) ROC CURVE
# --------------------------------------------------

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend()
    plt.tight_layout()

    file_path = f"{OUTPUT_DIR}/roc_curve_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"ROC Curve gespeichert unter: {file_path}")


# --------------------------------------------------
# 3) FEATURE IMPORTANCE
# --------------------------------------------------

def plot_feature_importance(model, X, model_name, top_n=15):
    if hasattr(model, "coef_"):
        importance = pd.Series(
            model.coef_[0],
            index=X.columns
        ).sort_values(key=abs, ascending=False)

    elif hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

    else:
        print(f"No feature importance for {model_name}")
        return

    importance = importance.head(top_n)

    plt.figure(figsize=(7, 5))
    importance[::-1].plot(kind="barh")
    plt.title(f"Top {top_n} Features – {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()

    file_path = f"{OUTPUT_DIR}/feature_importance_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"Feature Importance gespeichert unter: {file_path}")


# --------------------------------------------------
# 4) TOP-X / LIFT ANALYSIS
# --------------------------------------------------

def topx_lift_analysis(y_true, y_prob, model_name, top_percents=None):
    """
    Top-X Capture Rate & Lift Analyse inkl. Visualisierung
    """

    if top_percents is None:
        top_percents = [1, 5, 10, 20, 30]

    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob
    }).sort_values("y_prob", ascending=False)

    total_positives = df["y_true"].sum()
    n = len(df)

    results = []

    for p in top_percents:
        cutoff = int(np.ceil(n * p / 100))
        top_df = df.iloc[:cutoff]

        found = top_df["y_true"].sum()
        capture_rate = found / total_positives if total_positives > 0 else 0
        lift = capture_rate / (p / 100)

        results.append({
            "Top_%": p,
            "Checked_Records": cutoff,
            "Billing_Found": int(found),
            "Capture_Rate": capture_rate,
            "Lift": lift
        })

    results_df = pd.DataFrame(results)

    print(f"\nTop-X / Lift Analysis – {model_name}")
    print(results_df)

    _plot_topx_curve(results_df, model_name)


def _plot_topx_curve(results_df, model_name):
    plt.figure(figsize=(6, 4))

    x = results_df["Top_%"]
    y_model = results_df["Capture_Rate"]
    y_random = x / 100

    plt.plot(x, y_model, marker="o", label="Model")
    plt.plot(x, y_random, linestyle="--", label="Random")

    plt.xlabel("Top X % of invoices reviewed")
    plt.ylabel("Share of Billing cases captured")
    plt.title(f"Top-X Capture Curve – {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    file_path = f"{OUTPUT_DIR}/topx_lift_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"Top-X / Lift Curve gespeichert unter: {file_path}")


# --------------------------------------------------
# 5) FULL EVALUATION (inkl. Top-X / Lift)
# --------------------------------------------------

def full_visual_evaluation(model, X, y, model_name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    print(f"\n=== {model_name} ===")
    print("ROC-AUC:", roc_auc_score(y, y_prob))
    print(classification_report(y, y_pred))

    plot_confusion_matrix(y, y_pred, model_name)
    plot_roc_curve(y, y_prob, model_name)

    # 🔹 NEU: Top-X / Lift
    topx_lift_analysis(y, y_prob, model_name)
