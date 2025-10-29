import os
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision("high")

# ============================================================
#   Simple MLP
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
#   Train / Eval utilities
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


def fedavg_weighted(models, sizes):
    total = float(sum(sizes))
    new_state = deepcopy(models[0].state_dict())
    for k in new_state.keys():
        new_state[k] = sum(m.state_dict()[k] * (sz / total) for m, sz in zip(models, sizes))
    return new_state


# ============================================================
#   Main Trainer
# ============================================================
class FedAvgTrainer:
    def __init__(self, workdir="FL_data",
                 hidden_dim=64, batch_size=32,
                 local_epochs=5, rounds=10,
                 lr=5e-4, weight_decay=1e-4,
                 patience=5, seed=42, device=None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.rounds = rounds
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # placeholders
        self.input_dim = None
        self.output_dim = None
        self.client_loaders = []
        self.server_loaders = None
        self.global_model = None

        # metrics
        self.hist_val_loss = []
        self.hist_val_acc = []
        self.hist_server_train_loss = []
        self.hist_server_train_acc = []
        self.client_hist = []

    # ----------------------------------------------------
    # Merge train/test datasets
    # ----------------------------------------------------
    def merge_train_test(self):
        df1_train = pd.read_csv('data_hn_clinical_train.csv')
        df2_train = pd.read_csv('data_hn_ct_train.csv')
        df3_train = pd.read_csv('data_hn_pt_train.csv')
        df1_test  = pd.read_csv('data_hn_clinical_test.csv')
        df2_test  = pd.read_csv('data_hn_ct_test.csv')
        df3_test  = pd.read_csv('data_hn_pt_test.csv')

        merged_train_df = df1_train.merge(df2_train, on=['PatientID', 'CenterID', 'Outcome'], how='inner') \
                                   .merge(df3_train, on=['PatientID', 'CenterID', 'Outcome'], how='inner')
        merged_test_df = df1_test.merge(df2_test, on=['PatientID', 'CenterID', 'Outcome'], how='inner') \
                                 .merge(df3_test, on=['PatientID', 'CenterID', 'Outcome'], how='inner')
        return merged_train_df, merged_test_df

    # ----------------------------------------------------
    # Build loaders from merged data
    # ----------------------------------------------------
    def prepare(self):
        merged_train_df, merged_test_df = self.merge_train_test()

        # numeric only
        feat_cols = merged_train_df.select_dtypes(include=[np.number]).columns.drop("Outcome")
        X_train = merged_train_df[feat_cols].values
        y_train = merged_train_df["Outcome"].astype(int).values
        X_test  = merged_test_df[feat_cols].values
        y_test  = merged_test_df["Outcome"].astype(int).values

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        self.input_dim = X_train.shape[1]
        self.output_dim = len(np.unique(y_train))

        # create DataLoaders
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_val, dtype=torch.float32)
        y_val_t   = torch.tensor(y_val, dtype=torch.long)
        X_test_t  = torch.tensor(X_test, dtype=torch.float32)
        y_test_t  = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=self.batch_size, shuffle=False)
        self.server_loaders = (train_loader, val_loader, test_loader)

        # simulate 5 clients with different sizes
        dataset_size = len(X_train_t)
        proportions = [0.4, 0.25, 0.15, 0.1, 0.1]
        split_sizes = [int(p * dataset_size) for p in proportions]
        split_sizes[-1] += dataset_size - sum(split_sizes)
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]
        X_splits = torch.split(X_train_t, split_sizes)
        y_splits = torch.split(y_train_t, split_sizes)

        print("\n📊 Client data distribution:")
        for i, s in enumerate(split_sizes, 1):
            print(f"Client {i}: {s} samples ({s / dataset_size:.1%})")

        self.client_loaders = [
            DataLoader(TensorDataset(X_splits[i], y_splits[i]), batch_size=self.batch_size, shuffle=True)
            for i in range(len(split_sizes))
        ]
        self.client_sizes = split_sizes

        self.global_model = MLP(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)

    # ----------------------------------------------------
    # Federated training
    # ----------------------------------------------------
    def run(self):
        if self.global_model is None:
            raise RuntimeError("Call prepare() before run().")
        criterion = nn.CrossEntropyLoss()
        best_val, stale = float("inf"), 0
        train_srv, val_srv, test_srv = self.server_loaders

        for rnd in range(1, self.rounds + 1):
            print(f"\n--- FedAvg Round {rnd}/{self.rounds} ---")
            client_models, rec = [], {"loss": [], "acc": []}

            for loader in self.client_loaders:
                m = MLP(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
                m.load_state_dict(self.global_model.state_dict(), strict=True)
                opt = torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.weight_decay)

                for _ in range(self.local_epochs):
                    tr_loss, tr_acc = train_one_epoch(m, loader, opt, criterion, self.device)
                rec["loss"].append(tr_loss)
                rec["acc"].append(tr_acc)
                client_models.append(m)

            self.client_hist.append(rec)

            # weighted aggregation
            new_state = fedavg_weighted(client_models, self.client_sizes)
            self.global_model.load_state_dict(new_state)

            tr_loss, tr_acc = evaluate(self.global_model, train_srv, criterion, self.device)[:2]
            va_loss, va_acc = evaluate(self.global_model, val_srv, criterion, self.device)[:2]
            self.hist_server_train_loss.append(tr_loss)
            self.hist_server_train_acc.append(tr_acc)
            self.hist_val_loss.append(va_loss)
            self.hist_val_acc.append(va_acc)

            print(f"Server Train -> loss {tr_loss:.4f} | acc {tr_acc:.4f}")
            print(f"Server Val   -> loss {va_loss:.4f} | acc {va_acc:.4f}")

            if va_loss < best_val - 1e-6:
                best_val, stale = va_loss, 0
                best_state = deepcopy(self.global_model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    print("Early stopping.")
                    break

        if "best_state" in locals():
            self.global_model.load_state_dict(best_state)

        te_loss, te_acc, te_f1, y_true, y_pred = evaluate(self.global_model, test_srv, criterion, self.device)
        print(f"\n✅ Final Test -> loss {te_loss:.4f} | acc {te_acc:.4f} | f1 {te_f1:.4f}")
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
        print("\nClassification report:\n", classification_report(y_true, y_pred))
        return te_loss, te_acc, te_f1

    def plot_server_curves(self):
        r = range(1, len(self.hist_val_loss) + 1)
        plt.figure(figsize=(9, 5))
        plt.plot(r, self.hist_val_loss, label="Val Loss")
        plt.plot(r, self.hist_val_acc, label="Val Acc")
        plt.plot(r, self.hist_server_train_loss, label="Train Loss (server)")
        plt.plot(r, self.hist_server_train_acc, label="Train Acc (server)")
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title("Server metrics across FedAvg rounds")
        plt.grid(True)
        plt.legend()
        plt.show()


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    trainer = FedAvgTrainer(rounds=15, local_epochs=5)
    trainer.prepare()
    trainer.run()
    trainer.plot_server_curves()
