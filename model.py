"""
model.py
--------
Defines the FraudDetector class, which encapsulates:
  1. Data loading and preprocessing (scaling, train/test split)
  2. Class imbalance handling via SMOTE
  3. Model training (Random Forest or XGBoost)
  4. Saving and loading trained models

Usage:
    from model import FraudDetector
    detector = FraudDetector(model_type="xgboost")
    detector.load_data("creditcard.csv")
    detector.preprocess()
    detector.train()
    detector.save_model("fraud_model.joblib")
"""

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class FraudDetector:
    """
    End-to-end fraud detection pipeline.

    Handles preprocessing, SMOTE oversampling, model training,
    and persistence for the Credit Card Fraud Detection task.
    """

    # Columns that need standard scaling (not PCA features)
    SCALE_COLS = ["Time", "Amount"]

    def __init__(
        self,
        model_type: str = "xgboost",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialise the detector.

        Parameters
        ----------
        model_type    : 'xgboost' or 'random_forest'
        test_size     : Fraction of data reserved for testing (default 20%).
        random_state  : Seed for reproducibility.
        """
        if model_type not in ("xgboost", "random_forest"):
            raise ValueError("model_type must be 'xgboost' or 'random_forest'")

        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state

        # Set during preprocessing
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Set during training
        self.model = None

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load the dataset from a CSV file.

        Parameters
        ----------
        csv_path : Path to the creditcard.csv file.

        Returns
        -------
        pd.DataFrame : Raw dataframe.
        """
        print(f"Loading data from '{csv_path}'...")
        df = pd.read_csv(csv_path)

        # Basic validation
        required_cols = ["Time", "Amount", "Class"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"  Loaded {len(df):,} rows | "
              f"Fraud rate: {df['Class'].mean():.4%}")
        self.df = df
        return df

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self) -> None:
        """
        Prepare features for training:
          - Scale 'Time' and 'Amount' (V1-V28 are already PCA-transformed)
          - Split into train / test sets (stratified to preserve fraud ratio)
        """
        print("\nPreprocessing...")

        df = self.df.copy()

        # Scale Time and Amount; leave V1-V28 as-is
        df[self.SCALE_COLS] = self.scaler.fit_transform(df[self.SCALE_COLS])

        # Separate features and target
        X = df.drop(columns=["Class"])
        y = df["Class"]
        self.feature_names = list(X.columns)

        # Stratified split — preserves the 0.17% fraud ratio in both sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print(f"  Train size: {len(self.X_train):,} "
              f"(fraud: {self.y_train.sum():,})")
        print(f"  Test size : {len(self.X_test):,} "
              f"(fraud: {self.y_test.sum():,})")

    # ------------------------------------------------------------------
    # 3. SMOTE — handle class imbalance
    # ------------------------------------------------------------------

    def _apply_smote(self) -> tuple:
        """
        Apply SMOTE to the training data only.

        SMOTE (Synthetic Minority Over-sampling Technique) generates
        synthetic fraud samples so the model sees a balanced training set.
        IMPORTANT: SMOTE is applied ONLY to training data, never test data.

        Returns
        -------
        X_resampled, y_resampled : Balanced arrays.
        """
        print("\nApplying SMOTE to training data...")

        smote = SMOTE(
            sampling_strategy=0.1,  # Oversample fraud to 10% of majority class
            random_state=self.random_state,
            k_neighbors=5,
        )

        X_res, y_res = smote.fit_resample(self.X_train, self.y_train)

        fraud_count = int(y_res.sum())
        legit_count = int((y_res == 0).sum())
        print(f"  After SMOTE — Legit: {legit_count:,} | Fraud: {fraud_count:,}")

        return X_res, y_res

    # ------------------------------------------------------------------
    # 4. Model building
    # ------------------------------------------------------------------

    def _build_model(self):
        """
        Instantiate the chosen classifier with fraud-optimised hyperparameters.

        Both models use class_weight / scale_pos_weight to further penalise
        missed fraud cases on top of SMOTE resampling.

        Returns
        -------
        Unfitted sklearn-compatible classifier.
        """
        if self.model_type == "xgboost":
            # scale_pos_weight ~ ratio of negatives to positives before SMOTE
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=10,   # Extra weight on fraud class
                use_label_encoder=False,
                eval_metric="aucpr",   # Area under Precision-Recall (better for imbalance)
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight={0: 1, 1: 10},  # Penalise missing fraud
                random_state=self.random_state,
                n_jobs=-1,
            )

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Full training pipeline:
          1. Apply SMOTE to balance training data
          2. Build model
          3. Fit model on resampled training data
        """
        if self.X_train is None:
            raise RuntimeError("Call preprocess() before train().")

        # Resample training data with SMOTE
        X_res, y_res = self._apply_smote()

        # Build and fit model
        self.model = self._build_model()
        print(f"\nTraining {self.model_type.replace('_', ' ').title()}...")
        self.model.fit(X_res, y_res)
        print("  Training complete.")

    # ------------------------------------------------------------------
    # 6. Prediction helpers
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (0 or 1) for input X.

        Parameters
        ----------
        X : Feature array (already scaled).

        Returns
        -------
        np.ndarray : Binary predictions.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probability for input X.

        Parameters
        ----------
        X : Feature array (already scaled).

        Returns
        -------
        np.ndarray : Probability scores for the fraud class (column 1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # 7. Model persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str = "fraud_model.joblib") -> None:
        """
        Save the trained model and scaler to disk.

        Parameters
        ----------
        path : Output file path (e.g. 'fraud_model.joblib').
        """
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }
        joblib.dump(payload, path)
        print(f"\nModel saved to '{path}'")

    @classmethod
    def load_model(cls, path: str) -> "FraudDetector":
        """
        Load a previously saved FraudDetector from disk.

        Parameters
        ----------
        path : Path to the saved .joblib file.

        Returns
        -------
        FraudDetector instance ready for prediction.
        """
        payload = joblib.load(path)
        detector = cls(model_type=payload["model_type"])
        detector.model = payload["model"]
        detector.scaler = payload["scaler"]
        detector.feature_names = payload["feature_names"]
        print(f"Model loaded from '{path}'")
        return detector
