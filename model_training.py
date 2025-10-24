# model_training.py
"""
ModelTraining class for the eHealth project.
- Loads Clinical / CT / PET CSVs
- Applies cleaning via functions from clean.py
- Merges datasets on PatientID/CenterID/Outcome
- Prepares data (features/target, scaling, split)
- Trains baseline models (kNN by default)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Import cleaning utilities from local module
from Clean import (
    fix_structural_errors,
    check_duplicates,
    handle_missing_values,
    detect_and_remove_outliers,
)


class ModelTraining:
    #\"\"\"Encapsulates the full ML preparation workflow for the eHealth datasets.\"\"\"

    def __init__(self, path_clinical: str, path_ct: str, path_pt: str):
        self.path_clinical = path_clinical
        self.path_ct = path_ct
        self.path_pt = path_pt

        # Will be set later
        self.dfcli: pd.DataFrame | None = None
        self.dfct: pd.DataFrame | None = None
        self.dfpt: pd.DataFrame | None = None
        self.df_all: pd.DataFrame | None = None

        self.scaler: StandardScaler | None = None
        self.model = None

    # -------------------------------
    # Data loading & cleaning
    # -------------------------------
    def load_and_clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        #\"\"\"Load the three CSVs and apply the cleaning pipeline. Returns the three cleaned DataFrames.\"\"\"
        # Load
        dfcli = pd.read_csv(self.path_clinical)
        dfct = pd.read_csv(self.path_ct)
        dfpt = pd.read_csv(self.path_pt)

        # Structural fixes
        dfcli = fix_structural_errors(dfcli, "clinical")
        dfct = fix_structural_errors(dfct, "ct")
        dfpt = fix_structural_errors(dfpt, "pt")

        # Duplicates (by patient)
        dfcli = check_duplicates(dfcli, "Clinical", subset=["PatientID"])
        dfct = check_duplicates(dfct, "CT", subset=["PatientID"])
        dfpt = check_duplicates(dfpt, "PT", subset=["PatientID"])

        # Missing values
        dfcli = handle_missing_values(dfcli, "clinical")
        dfct = handle_missing_values(dfct, "ct")
        dfpt = handle_missing_values(dfpt, "pt")

        # Outliers (clinical only)
        dfcli = detect_and_remove_outliers(dfcli, "clinical")

        # Save to attributes
        self.dfcli, self.dfct, self.dfpt = dfcli, dfct, dfpt
        return dfcli, dfct, dfpt

    def merge_datasets(self, dfcli: pd.DataFrame, dfct: pd.DataFrame, dfpt: pd.DataFrame) -> pd.DataFrame:
        #\"\"\"Merge the three datasets on PatientID, CenterID and Outcome.\"\"\"
        df = dfcli.merge(dfct, on=["PatientID", "CenterID", "Outcome"])
        df = df.merge(dfpt, on=["PatientID", "CenterID", "Outcome"])
        self.df_all = df
        return df

    # -------------------------------
    # Features / target & scaling
    # -------------------------------
    def prepare_data(
        self, df_merged: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        #\"\"\"Split into X/y, standardize features, and create train/test sets.\"\"\"
        X = df_merged.drop(columns=["PatientID", "CenterID", "Outcome"])
        y = df_merged["Outcome"].astype(int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    # -------------------------------
    # Baseline model: kNN
    # -------------------------------
    def train_knn(self, X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 5) -> KNeighborsClassifier:
        #\"\"\"Train a baseline kNN classifier.\"\"\"
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        self.model = knn
        return knn

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        #\"\"\"Evaluate the current model on the test set and return metrics.\"\"\"
        if self.model is None:
            raise RuntimeError("No model trained yet. Call train_*() first.")

        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # ROC AUC (requires predict_proba or decision_function)
        try:
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(X_test)[:, 1]
            else:
                # Fallback to decision function if available
                y_score = self.model.decision_function(X_test)
            metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        except Exception:
            metrics["roc_auc"] = float("nan")

        print("Evaluation metrics:", metrics)
        print("\nClassification report:\n", classification_report(y_test, y_pred))

        return metrics


