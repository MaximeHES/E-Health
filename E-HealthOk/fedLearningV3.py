import os
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("high")

# ============================================================
# Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 64
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
ROUNDS = 15
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, p_drop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# ============================================================
# Train / Eval utilities
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct += (out.argmax(1) == y).sum().item()
        n += len(X)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    y_true, y_pred = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(X)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += len(X)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    acc = correct / n
    f1 = f1_score(y_true, y_pred, average="weighted")
    return total_loss / n, acc, f1, y_true, y_pred

# ============================================================
# FedAvg aggregation
# ============================================================
def fedavg_weighted(models, sizes):
    total = float(sum(sizes))
    new_state = deepcopy(models[0].state_dict())
    for k in new_state.keys():
        new_state[k] = sum(m.state_dict()[k] * (sz / total) for m, sz in zip(models, sizes))
    return new_state

def fedavg_unweighted(models):
    new_state = deepcopy(models[0].state_dict())
    for k in new_state.keys():
        new_state[k] = sum(m.state_dict()[k] for m in models) / len(models)
    return new_state

# ============================================================
# Data loading
# ============================================================
def load_cleaned_data():
    train_path = "CleanedData/cleaned_train.csv"
    test_path = "CleanedData/cleaned_test.csv"
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Cleaned data not found. Please run TrainModel.py first.")
    return pd.read_csv(train_path), pd.read_csv(test_path)

def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def build_clients(merged_train_df, n_clients=5, proportions=None):
    """Simulate clients from the cleaned training data."""
    feat_cols = merged_train_df.columns.drop(['Outcome', 'PatientID', 'CenterID'])
    X = merged_train_df[feat_cols].values
    y = merged_train_df['Outcome'].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # server loaders (for validation)
    train_loader_srv = make_loader(X_train, y_train)
    val_loader_srv = make_loader(X_val, y_val, shuffle=False)

    # client splits
    if proportions is None:
        proportions = [0.4, 0.25, 0.15, 0.1, 0.1]
    sizes = [int(p * len(X_train)) for p in proportions]
    sizes[-1] += len(X_train) - sum(sizes)

    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    client_loaders, client_sizes = [], []
    start = 0
    for s in sizes:
        Xi, yi = X_train[start:start + s], y_train[start:start + s]
        start += s
        client_loaders.append(make_loader(Xi, yi))
        client_sizes.append(len(yi))

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y))
    return client_loaders, client_sizes, (train_loader_srv, val_loader_srv), scaler, feat_cols, input_dim, output_dim

# ============================================================
# Federated Learning
# ============================================================
def run_fedavg(weighted=True):
    print(f"\n=== Running Federated Learning ({'Weighted' if weighted else 'Unweighted'} FedAvg) ===")

    merged_train_df, merged_test_df = load_cleaned_data()
    client_loaders, client_sizes, (train_srv, val_srv), scaler, feat_cols, input_dim, output_dim = build_clients(merged_train_df)
    X_test = merged_test_df[feat_cols].values
    y_test = merged_test_df['Outcome'].astype(int).values
    X_test = scaler.transform(X_test)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    # initialize model
    global_model = MLP(input_dim, output_dim, HIDDEN_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_val, stale = float('inf'), 0
    hist_val_loss, hist_val_acc, hist_train_loss, hist_train_acc = [], [], [], []

    for rnd in range(1, ROUNDS + 1):
        print(f"\n--- Round {rnd}/{ROUNDS} ---")
        client_models = []
        for loader in client_loaders:
            m = MLP(input_dim, output_dim, HIDDEN_DIM).to(DEVICE)
            m.load_state_dict(global_model.state_dict())
            opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            for _ in range(LOCAL_EPOCHS):
                tr_loss, tr_acc = train_one_epoch(m, loader, opt, criterion, DEVICE)
            client_models.append(m)

        # aggregation
        if weighted:
            new_state = fedavg_weighted(client_models, client_sizes)
        else:
            new_state = fedavg_unweighted(client_models)
        global_model.load_state_dict(new_state)

        tr_loss, tr_acc, _, _, _ = evaluate(global_model, train_srv, criterion, DEVICE)
        va_loss, va_acc, _, _, _ = evaluate(global_model, val_srv, criterion, DEVICE)
        hist_train_loss.append(tr_loss)
        hist_train_acc.append(tr_acc)
        hist_val_loss.append(va_loss)
        hist_val_acc.append(va_acc)

        print(f"Server Train -> loss {tr_loss:.4f}, acc {tr_acc:.4f}")
        print(f"Server Val   -> loss {va_loss:.4f}, acc {va_acc:.4f}")

        if va_loss < best_val:
            best_val, stale = va_loss, 0
            best_state = deepcopy(global_model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping.")
                break

    global_model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, y_true, y_pred = evaluate(global_model, test_loader, criterion, DEVICE)
    print(f"\n✅ Final Test -> loss {te_loss:.4f} | acc {te_acc:.4f} | f1 {te_f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # plot
    r = range(1, len(hist_val_loss) + 1)
    plt.figure(figsize=(9,5))
    plt.plot(r, hist_val_loss, label="Val Loss")
    plt.plot(r, hist_val_acc, label="Val Acc")
    plt.plot(r, hist_train_loss, label="Train Loss")
    plt.plot(r, hist_train_acc, label="Train Acc")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title(f"Server Metrics Across FedAvg Rounds ({'Weighted' if weighted else 'Unweighted'})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return te_acc, te_f1

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    acc_w, f1_w = run_fedavg(weighted=True)
    acc_u, f1_u = run_fedavg(weighted=False)

    print("\n=== Comparison ===")
    print(f"FedAvg Weighted   -> Acc: {acc_w:.4f}, F1: {f1_w:.4f}")
    print(f"FedAvg Unweighted -> Acc: {acc_u:.4f}, F1: {f1_u:.4f}")
