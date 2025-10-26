# model_training_torch.py
"""
PyTorch MLP model for the eHealth project.
- Loads Clinical / CT / PET CSVs
- Cleans data using clean.py
- Merges datasets
- Prepares data tensors
- Trains and evaluates a simple MLP
"""

import pandas as pd
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from Clean import (
    fix_structural_errors,
    check_duplicates,
    handle_missing_values,
    detect_and_remove_outliers,
)

# -------------------------------
# Define MLP model
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary output
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# ModelTraining class using PyTorch
# -------------------------------
class ModelTrainingTorch:
    def __init__(self, path_clinical: str, path_ct: str, path_pt: str):
        self.path_clinical = path_clinical
        self.path_ct = path_ct
        self.path_pt = path_pt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = None

    # -------------------------------
    # Data loading & cleaning
    # -------------------------------
    def load_and_clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dfcli = pd.read_csv(self.path_clinical)
        dfct = pd.read_csv(self.path_ct)
        dfpt = pd.read_csv(self.path_pt)

        # Cleaning pipeline
        dfcli = fix_structural_errors(dfcli, "clinical")
        dfct = fix_structural_errors(dfct, "ct")
        dfpt = fix_structural_errors(dfpt, "pt")

        dfcli = check_duplicates(dfcli, "Clinical", subset=["PatientID"])
        dfct = check_duplicates(dfct, "CT", subset=["PatientID"])
        dfpt = check_duplicates(dfpt, "PT", subset=["PatientID"])

        dfcli = handle_missing_values(dfcli, "clinical")
        dfct = handle_missing_values(dfct, "ct")
        dfpt = handle_missing_values(dfpt, "pt")

        dfcli = detect_and_remove_outliers(dfcli, "clinical")

        return dfcli, dfct, dfpt

    def merge_datasets(self, dfcli: pd.DataFrame, dfct: pd.DataFrame, dfpt: pd.DataFrame) -> pd.DataFrame:
        df = dfcli.merge(dfct, on=["PatientID", "CenterID", "Outcome"])
        df = df.merge(dfpt, on=["PatientID", "CenterID", "Outcome"])
        return df

    # -------------------------------
    # Prepare tensors
    # -------------------------------
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        X = df.drop(columns=["PatientID", "CenterID", "Outcome"]).values.astype(np.float32)
        y = df["Outcome"].values.astype(np.float32).reshape(-1, 1)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)

        n_train = int((1 - test_size) * len(dataset))
        n_test = len(dataset) - n_train
        train_set, test_set = random_split(dataset, [n_train, n_test])
        return train_set, test_set

    # -------------------------------
    # Training
    # -------------------------------
    def train_mlp(self, train_set, test_set, epochs=50, lr=1e-3, batch_size=32):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        input_dim = train_set.dataset.tensors[0].shape[1]
        model = MLP(input_dim).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

        self.model = model
        self.evaluate(test_loader)
        return model

    # -------------------------------
    # Evaluation
    # -------------------------------
    def evaluate(self, test_loader):
        model = self.model
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                preds = model(X_batch)
                preds = (preds.cpu() > 0.5).float()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = (y_true == y_pred).mean()
        print(f"\nTest Accuracy: {acc:.3f}")
