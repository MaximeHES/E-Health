# ===========================================
#   E-Health MLP Training Pipeline
#   Merge Clinical + CT + PT, Clean, Train, Evaluate
# ===========================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# -------------------------------
# Hyperparameters
# -------------------------------
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 10


# -------------------------------
# Cleaning helpers
# -------------------------------
def fix_structural_errors(df, source="clinical"):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    if source.lower() == "clinical":
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
        for col in ["Tobacco", "Alcohol", "Surgery", "Chemotherapy", "Outcome"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)
        if "Performance status" in df.columns:
            df.rename(columns={"Performance status": "Performance_status"}, inplace=True)
    else:
        for col in [c for c in df.columns if c not in {"PatientID", "CenterID"}]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace(["?", "NA", "N/A", "na", "-", "None", "null"], np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df


def handle_missing_values(df, source="clinical"):
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
# Safe merge on PatientID + CenterID
# -------------------------------
def merge_datasets_safe(cli_path, ct_path, pt_path):
    cli = handle_missing_values(fix_structural_errors(pd.read_csv(cli_path), "clinical"), "clinical")
    ct  = handle_missing_values(fix_structural_errors(pd.read_csv(ct_path),  "ct"), "ct")
    pt  = handle_missing_values(fix_structural_errors(pd.read_csv(pt_path),  "pt"), "pt")

    # Choose keys
    keys = ["PatientID", "CenterID","Outcome"] if all(k in df.columns for df in [cli, ct, pt] for k in ["PatientID", "CenterID","Outcome"]) else ["PatientID"]

    # Remove duplicates per key
    for name, df in [("clinical", cli), ("ct", ct), ("pt", pt)]:
        dups = df.duplicated(subset=keys).sum()
        if dups:
            print(f"⚠️ {name} has {dups} duplicate rows for keys {keys}. Dropping duplicates.")
            df.drop_duplicates(subset=keys, inplace=True)

    # Merge
    merged = cli.merge(ct, on=keys, how="inner", suffixes=("", "_CT")) \
                .merge(pt, on=keys, how="inner", suffixes=("", "_PT"))

    # Keep Outcome from clinical
    if "Outcome" not in merged.columns and "Outcome" in cli.columns:
        merged = merged.merge(cli[keys + ["Outcome"]], on=keys, how="left")

    return merged, keys


# -------------------------------
# MLP Model
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -------------------------------
# Training and evaluation loops
# -------------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# -------------------------------
# Data preparation
# -------------------------------

#
def prepare_train_test():
    base = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(base, "Ressources")

    cli_train = os.path.join(res, "data_hn_clinical_train.csv")
    ct_train  = os.path.join(res, "data_hn_ct_train.csv")
    pt_train  = os.path.join(res, "data_hn_pt_train.csv")
    cli_test  = os.path.join(res, "data_hn_clinical_test.csv")
    ct_test   = os.path.join(res, "data_hn_ct_test.csv")
    pt_test   = os.path.join(res, "data_hn_pt_test.csv")

    print("🔗 Merging training datasets...")
    train_df, merge_keys = merge_datasets_safe(cli_train, ct_train, pt_train)
    print("🔗 Merging test datasets...")
    test_df, _ = merge_datasets_safe(cli_test, ct_test, pt_test)

    # Encode gender to int
    for df in (train_df, test_df):
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"M": 0, "F": 1}).fillna(0)

    # Drop unwanted columns
    leaky_cols = ["CenterID"]  # can remove later if desired
    X_train = train_df.drop(columns=merge_keys + ["Outcome"] + leaky_cols, errors="ignore")
    y_train = train_df["Outcome"].astype(int)
    X_test  = test_df.drop(columns=merge_keys + ["Outcome"] + leaky_cols, errors="ignore")
    y_test  = test_df["Outcome"].astype(int)

    # One-hot encode categorical cols & align
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test, drop_first=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    #Just to check how many ligns we have
    print(f"Train dataset rows: {len(train_df)}")
    print(f"Test dataset rows: {len(test_df)}")

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val.values, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_t.shape[1]
    output_dim = len(y_train.unique())
    return train_loader, val_loader, test_loader, input_dim, output_dim


# -------------------------------
# Main training routine
# -------------------------------
def main():
    train_loader, val_loader, test_loader, INPUT_DIM, OUTPUT_DIM = prepare_train_test()
    print(f"✅ Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")

    model = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n🚀 Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # --- Test evaluation ---
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Detailed metrics
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            preds = torch.argmax(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()
