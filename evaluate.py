"""
evaluate.py
-----------
Evaluation utilities for the fraud detection model.

Generates:
  1. Classification report  (precision, recall, F1 for each class)
  2. Confusion matrix        (with percentage annotations)
  3. Precision-Recall curve  (more meaningful than ROC for imbalanced data)
  4. Feature importance plot (top 20 features)

Usage:
    from model import FraudDetector
    from evaluate import evaluate_model

    evaluate_model(detector)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _set_style() -> None:
    """Apply a clean, minimal matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })


# ---------------------------------------------------------------------------
# 1. Classification report
# ---------------------------------------------------------------------------

def print_classification_report(y_true, y_pred) -> None:
    """
    Print a detailed classification report.

    Focuses on Class 1 (fraud) metrics:
      - Precision : of all predicted frauds, how many were real?
      - Recall    : of all real frauds, how many did we catch?
      - F1-score  : harmonic mean of precision and recall

    Parameters
    ----------
    y_true : Ground-truth labels.
    y_pred : Predicted labels.
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate (0)", "Fraud (1)"],
        digits=4,
    ))


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path: str = None) -> None:
    """
    Plot a confusion matrix with raw counts and row-normalised percentages.

    Cells explained:
      TN (top-left)  : Legitimate correctly identified
      FP (top-right) : Legitimate wrongly flagged as fraud   (false alarm)
      FN (bot-left)  : Fraud missed by the model             (dangerous!)
      TP (bot-right) : Fraud correctly caught

    Parameters
    ----------
    y_true    : Ground-truth labels.
    y_pred    : Predicted labels.
    save_path : If provided, saves the figure to this path.
    """
    _set_style()

    cm = confusion_matrix(y_true, y_pred)
    # Row-normalise to show recall per class
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    labels = [
        [f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_pct,
        annot=labels,
        fmt="",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row %"},
    )

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Legitimate", "Fraud"], fontsize=11)
    ax.set_yticklabels(["Legitimate", "Fraud"], fontsize=11, rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix saved to '{save_path}'")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Precision-Recall curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(y_true, y_scores, save_path: str = None) -> None:
    """
    Plot the Precision-Recall curve.

    Why PR instead of ROC?
    ----------------------
    ROC curves can look optimistic for highly imbalanced datasets because
    the large number of true negatives inflates the FPR denominator.
    PR curves focus only on the minority class (fraud) and are more
    informative when false negatives (missed fraud) are costly.

    A high area under the PR curve (AUPRC) means the model is good at
    ranking fraud higher than legitimate transactions.

    Parameters
    ----------
    y_true   : Ground-truth labels.
    y_scores : Predicted probability scores for the fraud class.
    save_path: If provided, saves the figure to this path.
    """
    _set_style()

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Baseline = random classifier at the fraud rate
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="#1a6cc4", lw=2,
            label=f"Model (AUPRC = {auprc:.4f})")
    ax.axhline(baseline, color="#e63946", lw=1.5, linestyle="--",
               label=f"Random baseline ({baseline:.4f})")

    # Mark operating points at common thresholds
    for thr in [0.3, 0.5, 0.7]:
        idx = np.searchsorted(thresholds, thr)
        if idx < len(precision) - 1:
            ax.scatter(recall[idx], precision[idx], s=60, zorder=5)
            ax.annotate(
                f"t={thr}",
                (recall[idx], precision[idx]),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=9,
            )

    ax.set_xlabel("Recall (Fraud caught / All fraud)", fontsize=12)
    ax.set_ylabel("Precision (True fraud / All flagged)", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  PR curve saved to '{save_path}'")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             save_path: str = None) -> None:
    """
    Plot the top-N most important features from the trained model.

    Parameters
    ----------
    model        : Trained XGBClassifier or RandomForestClassifier.
    feature_names: List of feature column names.
    top_n        : Number of top features to display.
    save_path    : If provided, saves the figure to this path.
    """
    _set_style()

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1a6cc4" if v > np.median(values) else "#6baed6" for v in values]
    ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")

    ax.set_xlabel("Importance score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Feature importance plot saved to '{save_path}'")
    plt.show()


# ---------------------------------------------------------------------------
# Master evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(detector, save_plots: bool = True) -> dict:
    """
    Run the full evaluation suite on the test set.

    Steps:
      1. Generate predictions and probability scores
      2. Print classification report
      3. Plot confusion matrix
      4. Plot Precision-Recall curve
      5. Plot feature importances

    Parameters
    ----------
    detector   : A trained FraudDetector instance.
    save_plots : Whether to save plots as PNG files.

    Returns
    -------
    dict : {'y_pred': ..., 'y_scores': ..., 'auprc': ...}
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)

    y_pred = detector.predict(detector.X_test)
    y_scores = detector.predict_proba(detector.X_test)
    y_true = detector.y_test.values

    # 1. Classification report
    print_classification_report(y_true, y_pred)

    # 2. Confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred,
        save_path="confusion_matrix.png" if save_plots else None,
    )

    # 3. Precision-Recall curve
    print("Generating Precision-Recall curve...")
    plot_precision_recall_curve(
        y_true, y_scores,
        save_path="pr_curve.png" if save_plots else None,
    )

    # 4. Feature importances
    print("Generating feature importance plot...")
    plot_feature_importance(
        detector.model,
        detector.feature_names,
        save_path="feature_importance.png" if save_plots else None,
    )

    auprc = average_precision_score(y_true, y_scores)
    print(f"\nArea Under PR Curve (AUPRC): {auprc:.4f}")
    print("(Higher is better; random baseline ≈ fraud rate ≈ 0.0017)")

    return {"y_pred": y_pred, "y_scores": y_scores, "auprc": auprc}
