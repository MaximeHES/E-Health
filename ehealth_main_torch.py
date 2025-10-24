# -------------------------------
# 1. Imports
# -------------------------------
import os
import random
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------------
# 2. Cleaning Functions (embedded from Clean.py, simplified)
# -------------------------------
def fix_structural_errors(df: pd.DataFrame, source: str = "clinical") -> pd.DataFrame:
    df = df.copy()
    # Clean column names
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    if source.lower() == "clinical":
        # Text fields
        if "Gender" in df.columns:
            df["Gender"] = (
                df["Gender"]
                .astype(str).str.strip().str.upper()
                .replace({"MALE": "M", "FEMALE": "F"})
            )
        if "CenterID" in df.columns:
            df["CenterID"] = df["CenterID"].astype(str).str.strip().str.title()

        # Binary columns
        for col in ["Tobacco", "Alcohol", "Surgery", "Chemotherapy", "Outcome"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)

        # Numeric columns
        if "Performance status" in df.columns:
            df.rename(columns={"Performance status": "Performance_status"}, inplace=True)
        for col in ["Age", "Weight", "Performance_status"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # IDs
        for id_col in ["PatientID", "CenterID"]:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).str.strip()

    elif source.lower() in {"ct", "pt"}:
        # IDs
        if "PatientID" in df.columns:
            df["PatientID"] = df["PatientID"].astype(str).str.strip()
        if "CenterID" in df.columns:
            df["CenterID"] = df["CenterID"].astype(str).str.strip().str.title()

        # Cast all non-ID columns to numeric
        non_id_cols = [c for c in df.columns if c not in {"PatientID", "CenterID"}]
        df[non_id_cols] = df[non_id_cols].apply(pd.to_numeric, errors="coerce")

    # Replace common textual missing tokens with NaN
    df.replace(
        to_replace=["?", "NA", "N/A", "na", "Nan", "nan", "NaN", "None", "none", "null", "Null", "-"],
        value=np.nan,
        inplace=True
    )

    # Drop fully empty columns
    df.dropna(axis=1, how="all", inplace=True)

    return df


def handle_missing_values(df: pd.DataFrame, source: str = "clinical") -> pd.DataFrame:
    """
    Handle missing values:
    - Clinical: Tobacco/Alcohol -> 0 ; Performance_status -> median
    - CT/PT: median per column
    """
    df = df.copy()

    if source.lower() == "clinical":
        for col in ["Tobacco", "Alcohol"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        if "Performance_status" in df.columns:
            df["Performance_status"] = df["Performance_status"].fillna(df["Performance_status"].median())
    else:
        df = df.fillna(df.median(numeric_only=True))

    return df


# -------------------------------
# 3. PyTorch MLP Model
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary output
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# 4. Training utilities
# -------------------------------
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy().ravel()
            y_true.extend(yb.numpy().ravel())
            y_prob.extend(out)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    print("\n--- Evaluation Report ---")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
    return metrics

# -------------------------------
# 5bis. Data comparison utilities
# -------------------------------
def compare_before_after(before: pd.DataFrame, after: pd.DataFrame, name: str):
    """Display differences before and after cleaning."""
    print(f"\n🔍 Comparison for {name}")
    print("-" * 60)
    print(f"Shape before: {before.shape} | after: {after.shape}")

    # Missing values
    before_missing = before.isna().sum().sum()
    after_missing = after.isna().sum().sum()
    print(f"Missing values: before = {before_missing:,} | after = {after_missing:,}")

    # Column overview
    print("\nColumn summary (first 5):")
    summary = pd.DataFrame({
        "before_dtype": before.dtypes,
        "after_dtype": after.dtypes,
        "before_nan": before.isna().sum(),
        "after_nan": after.isna().sum(),
        "before_unique": before.nunique(),
        "after_unique": after.nunique()
    }).head(5)
    print(summary)

    # Changed columns
    changed_cols = [c for c in after.columns if c not in before.columns]
    if changed_cols:
        print("\n🆕 New columns after cleaning:", changed_cols[:5], "..." if len(changed_cols) > 5 else "")
    else:
        print("\nNo new columns introduced.")

# -------------------------------
# 5. Full Pipeline
# -------------------------------
def main():
    # --- Paths (robust, relative to this file) ---
    base = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(base, "Ressources")
    PATH_CLI = os.path.join(res, "data_hn_clinical_test.csv")
    PATH_CT  = os.path.join(res, "data_hn_ct_test.csv")
    PATH_PT  = os.path.join(res, "data_hn_pt_test.csv")

    print("📂 Loading CSV files...")
    if not (os.path.exists(PATH_CLI) and os.path.exists(PATH_CT) and os.path.exists(PATH_PT)):
        raise FileNotFoundError("One or more CSV files not found in the 'Ressources' folder. Please check paths.")

    dfcli = pd.read_csv(PATH_CLI)
    dfct  = pd.read_csv(PATH_CT)
    dfpt  = pd.read_csv(PATH_PT)

        # --- Load raw data ---
    dfcli_raw = pd.read_csv(PATH_CLI)
    dfct_raw  = pd.read_csv(PATH_CT)
    dfpt_raw  = pd.read_csv(PATH_PT)

    # --- Cleaning ---
    print("🧹 Cleaning data...")
    dfcli = handle_missing_values(fix_structural_errors(dfcli, "clinical"), "clinical")
    dfct  = handle_missing_values(fix_structural_errors(dfct,  "ct"),       "ct")
    dfpt  = handle_missing_values(fix_structural_errors(dfpt,  "pt"),       "pt")

    # --- Compare before/after ---
    compare_before_after(dfcli_raw, dfcli, "Clinical Data")
    compare_before_after(dfct_raw, dfct, "CT Data")
    compare_before_after(dfpt_raw, dfpt, "PT Data")

    # --- Merge datasets ---
    print("🔗 Merging datasets...")
    df = (
        dfcli.merge(dfct, on=["PatientID", "CenterID", "Outcome"], how="inner")
             .merge(dfpt, on=["PatientID", "CenterID", "Outcome"], how="inner")
    )
    print(f"Merged dataset: {df.shape} rows x {df.shape[1]} cols")

    # --- Encode categorical features ---
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"M": 0, "F": 1}).fillna(0)

    # --- Prepare features and labels ---
    X = df.drop(columns=["PatientID", "CenterID", "Outcome"])
    # One-hot encode remaining categorical columns if any
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    y_series = df["Outcome"].astype(int)
    y = y_series.values.astype(np.float32).reshape(-1, 1)

    # --- Train/Test split (stratified) ---
    X_train, X_test, y_train_1d, y_test_1d = train_test_split(
        X, y_series, test_size=0.2, stratify=y_series, random_state=SEED
    )

    # --- Scaling (fit on train only) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- Convert to tensors ---
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train_1d.values.astype(np.float32).reshape(-1, 1))
    y_test_t  = torch.tensor(y_test_1d.values.astype(np.float32).reshape(-1, 1))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)

    # --- Model setup ---
    input_dim = X_train_t.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    model = MLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # small L2

    # --- Training ---
    print("🚀 Training MLP model...")
    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss:.4f}")

    # --- Evaluation ---
    print("📊 Evaluating model...")
    _ = evaluate_model(model, test_loader, device)

    # --- Save model ---
    out_path = os.path.join(res, "mlp_ehealth_model.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\n✅ Training completed successfully. Model saved as '{out_path}'.")


# -------------------------------
# 6. Run
# -------------------------------
if __name__ == "__main__":
    main()
