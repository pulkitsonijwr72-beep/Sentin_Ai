"""
data_setup.py
-------------
Generates a synthetic Credit Card Fraud Detection dataset that mimics the
real Kaggle dataset structure:
  - 30 features: Time, V1-V28 (PCA components), Amount
  - Binary target: Class (0 = legitimate, 1 = fraud)
  - ~0.17% fraud rate (matching the real dataset imbalance)

Run this script first before anything else:
    python data_setup.py
"""

import numpy as np
import pandas as pd


def generate_fraud_dataset(
    n_samples: int = 284_807,
    fraud_rate: float = 0.0017,
    random_state: int = 42,
    output_path: str = "creditcard.csv",
) -> pd.DataFrame:
    """
    Generate a synthetic credit card transaction dataset.

    Parameters
    ----------
    n_samples     : Total number of transactions to generate.
    fraud_rate    : Fraction of transactions that are fraudulent.
    random_state  : Random seed for reproducibility.
    output_path   : File path to save the resulting CSV.

    Returns
    -------
    pd.DataFrame  : The generated dataset.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    print(f"Generating {n_samples:,} transactions "
          f"({n_legit:,} legitimate | {n_fraud:,} fraudulent)...")

    # --- Legitimate transactions ---
    # PCA features V1-V28: standard normal, slightly correlated
    legit_v = rng.standard_normal((n_legit, 28))

    # Time: spread evenly over 48 hours (in seconds)
    legit_time = rng.uniform(0, 172_800, n_legit)

    # Amount: log-normal distribution (most transactions are small)
    legit_amount = rng.lognormal(mean=3.0, sigma=1.5, size=n_legit)
    legit_amount = np.clip(legit_amount, 0.01, 25_000)

    # --- Fraudulent transactions ---
    # Fraud has different statistical signatures in PCA space
    fraud_v = rng.standard_normal((n_fraud, 28))
    # Shift certain components to simulate fraud patterns
    fraud_v[:, 0] -= 4.0   # V1: strongly negative in fraud
    fraud_v[:, 1] += 3.5   # V2: shifted positive
    fraud_v[:, 2] -= 2.5   # V3: shifted negative
    fraud_v[:, 3] += 2.0   # V4: slightly positive
    fraud_v[:, 9] -= 3.0   # V10: strongly negative
    fraud_v[:, 11] -= 3.5  # V12: strongly negative
    fraud_v[:, 13] -= 2.0  # V14: negative
    fraud_v[:, 15] += 2.5  # V16: positive
    fraud_v[:, 16] -= 2.0  # V17: negative

    # Fraud is spread throughout the day but slightly clustered at night
    fraud_time = rng.uniform(0, 172_800, n_fraud)

    # Fraud amounts: often smaller (card testing) or suspiciously round
    fraud_amount = rng.lognormal(mean=2.0, sigma=1.2, size=n_fraud)
    fraud_amount = np.clip(fraud_amount, 0.01, 5_000)

    # --- Combine into DataFrames ---
    v_cols = [f"V{i}" for i in range(1, 29)]

    df_legit = pd.DataFrame(legit_v, columns=v_cols)
    df_legit["Time"] = legit_time
    df_legit["Amount"] = legit_amount
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(fraud_v, columns=v_cols)
    df_fraud["Time"] = fraud_time
    df_fraud["Amount"] = fraud_amount
    df_fraud["Class"] = 1

    # Concatenate and shuffle
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Reorder columns to match real dataset
    cols = ["Time"] + v_cols + ["Amount", "Class"]
    df = df[cols]

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to '{output_path}'")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    print(f"Fraud rate: {df['Class'].mean():.4%}")

    return df


if __name__ == "__main__":
    generate_fraud_dataset()
