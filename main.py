"""
main.py
-------
Entry point for the Credit Card Fraud Detection project.

Pipeline:
  1. (Optional) Generate synthetic dataset via data_setup.py
  2. Load and preprocess data
  3. Train the fraud detection model (XGBoost with SMOTE)
  4. Evaluate on the held-out test set
  5. Save the trained model to disk

Run from your terminal:
    python main.py

To use Random Forest instead of XGBoost:
    Change MODEL_TYPE = "random_forest" below.
"""

import os
import time

from data_setup import generate_fraud_dataset
from evaluate import evaluate_model
from model import FraudDetector

# ---------------------------------------------------------------------------
# Configuration — edit these values to customise the run
# ---------------------------------------------------------------------------

DATA_PATH = "creditcard.csv"       # Path to the dataset CSV
MODEL_PATH = "fraud_model.joblib"  # Where to save the trained model
MODEL_TYPE = "xgboost"             # "xgboost" or "random_forest"
TEST_SIZE = 0.2                    # 20% of data held out for testing
RANDOM_STATE = 42                  # Seed for reproducibility
SAVE_PLOTS = True                  # Save evaluation plots as PNG files


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ensure_data_exists(path: str) -> None:
    """
    Check if the dataset file exists; generate a synthetic one if not.

    Parameters
    ----------
    path : Expected path to creditcard.csv
    """
    if not os.path.exists(path):
        print(f"'{path}' not found. Generating synthetic dataset...\n")
        generate_fraud_dataset(output_path=path)
    else:
        print(f"Found existing dataset at '{path}'.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the end-to-end fraud detection pipeline.
    """
    total_start = time.time()

    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — Production Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Ensure dataset is available
    # ------------------------------------------------------------------
    print("\n[Step 1/4] Checking dataset...")
    ensure_data_exists(DATA_PATH)

    # ------------------------------------------------------------------
    # Step 2: Load data and preprocess
    # ------------------------------------------------------------------
    print("\n[Step 2/4] Loading and preprocessing data...")
    detector = FraudDetector(
        model_type=MODEL_TYPE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    detector.load_data(DATA_PATH)
    detector.preprocess()

    # ------------------------------------------------------------------
    # Step 3: Train model
    # ------------------------------------------------------------------
    print("\n[Step 3/4] Training model...")
    t0 = time.time()
    detector.train()
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Evaluate on test set
    # ------------------------------------------------------------------
    print("\n[Step 4/4] Evaluating model...")
    results = evaluate_model(detector, save_plots=SAVE_PLOTS)

    # ------------------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------------------
    detector.save_model(MODEL_PATH)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model type  : {MODEL_TYPE}")
    print(f"  AUPRC       : {results['auprc']:.4f}")
    print(f"  Model saved : {MODEL_PATH}")
    if SAVE_PLOTS:
        print("  Plots saved : confusion_matrix.png, pr_curve.png, "
              "feature_importance.png")
    print(f"  Total time  : {total_elapsed:.1f}s")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Key metric reminder
    # ------------------------------------------------------------------
    print("""
Key metrics to focus on (NOT accuracy):
  - Recall (Fraud): % of real fraud cases caught — minimise missed fraud
  - Precision (Fraud): % of flagged cases that are real fraud — minimise false alarms
  - F1-score: balance between the two
  - AUPRC: area under Precision-Recall curve — overall ranking quality

A model with 99.83% accuracy can be achieved by predicting everything as
legitimate. Always check fraud-specific recall and precision.
    """)


if __name__ == "__main__":
    main()
