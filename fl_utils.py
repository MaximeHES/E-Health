"""
fl_utils.py — minimal utilities for Federated Learning on eHealth data.

Includes:
- Simple cleaning & merging (Clinical + CT + PT)
- Feature prep (one-hot, scaler)
- Client loaders per CenterID
- MLPBinary model (Sigmoid + BCELoss)
- Simple evaluation (loss + accuracy)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# -------------------------
# Cleaning & merging
# -------------------------
def _fix_structural_errors(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    if source == "clinical":
        if "Gender" in df.columns:
            df["Gender"] = (
                df["Gender"].astype(str).str.strip().str.upper()
                .replace({"MALE": "M", "FEMALE": "F"})
            )
        for col in ["Tobacco","Alcohol","Surgery","Chemotherapy","Outcome"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(0,1)
        if "Performance status" in df.columns:
            df = df.rename(columns={"Performance status": "Performance_status"})
        for col in ["Age","Weight","Performance_status"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        # ct / pt
        non_id = [c for c in df.columns if c not in {"PatientID","CenterID"}]
        df[non_id] = df[non_id].apply(pd.to_numeric, errors="coerce")
    df.replace(["?","NA","N/A","na","Nan","nan","NaN","None","none","null","Null","-"], np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df

def _handle_missing(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if source == "clinical":
        for col in ["Tobacco","Alcohol"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        if "Performance_status" in df.columns:
            df["Performance_status"] = df["Performance_status"].fillna(df["Performance_status"].median())
    else:
        df = df.fillna(df.median(numeric_only=True))
    return df

def load_and_merge(resources_dir: str, split: str = "train") -> pd.DataFrame:
    """Load Clinical/CT/PT CSVs for given split ('train' or 'test'), clean, and merge on PatientID/CenterID/Outcome."""
    assert split in {"train","test"}
    cli = pd.read_csv(os.path.join(resources_dir, f"data_hn_clinical_{split}.csv"))
    ct  = pd.read_csv(os.path.join(resources_dir, f"data_hn_ct_{split}.csv"))
    pt  = pd.read_csv(os.path.join(resources_dir, f"data_hn_pt_{split}.csv"))

    cli = _handle_missing(_fix_structural_errors(cli, "clinical"), "clinical")
    ct  = _handle_missing(_fix_structural_errors(ct,  "ct"),       "ct")
    pt  = _handle_missing(_fix_structural_errors(pt,  "pt"),       "pt")

    df = cli.merge(ct, on=["PatientID","CenterID","Outcome"], how="inner")             .merge(pt, on=["PatientID","CenterID","Outcome"], how="inner")

    if "Gender" in df.columns and df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"M":0,"F":1}).fillna(0)

    return df


# -------------------------
# Feature prep & loaders
# -------------------------
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler, pd.Index]:
    X = df.drop(columns=[c for c in ["PatientID","CenterID","Outcome"] if c in df.columns]).copy()
    X = pd.get_dummies(X, drop_first=True)
    y = df["Outcome"].astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    return X, Xs, y, scaler, X.columns

def align_and_scale(df_client: pd.DataFrame, scaler: StandardScaler, columns: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    Xc = df_client.drop(columns=[c for c in ["PatientID","CenterID","Outcome"] if c in df_client.columns]).copy()
    Xc = pd.get_dummies(Xc, drop_first=True)
    Xc = Xc.reindex(columns=columns, fill_value=0)
    Xc_scaled = scaler.transform(Xc.values)
    yc = df_client["Outcome"].astype(int).values
    return Xc_scaled, yc

def make_loader(Xs: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def make_client_loaders(df_train: pd.DataFrame, scaler: StandardScaler, columns: pd.Index, batch_size: int = 32):
    loaders, sizes = {}, {}
    for cid, dfc in df_train.groupby("CenterID"):
        Xc, yc = align_and_scale(dfc, scaler, columns)
        loaders[cid] = make_loader(Xc, yc, batch_size=batch_size, shuffle=True)
        sizes[cid]   = len(yc)
    return loaders, sizes


# -------------------------
# Model & evaluation
# -------------------------
class MLPBinary(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): 
        return self.net(x)

@torch.no_grad()
def evaluate_binary(model: nn.Module, loader: DataLoader, device: str = DEVICE):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_n = 0.0, 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        logits = model(xb)
        loss = criterion(logits, yb)
        proba = torch.sigmoid(logits)
        total_loss += loss.item() * xb.size(0)
        total_n    += xb.size(0)
        pred = (logits >= 0.5).float()
        correct += (pred.eq(yb)).sum().item()
    return total_loss / max(total_n,1), correct / max(total_n,1)
