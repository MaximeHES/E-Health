# ===========================================
#   E-Health Federated Learning (FedAvg)
#   Clean OOP version combining FLTrainOK + Trainer Class
# ===========================================

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# You can import your helper functions here if you already have them:
# from train import MLP, fix_structural_errors, handle_missing_values, merge_datasets_safe
# -------------------------------------------------------------------


# ------------------------------------------------
# Model definition
# ------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, p_drop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def train_local(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for _ in range(epochs):
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
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
    f1 = f1_score(y_true, y_pred, average='weighted')
    return total_loss / n, acc, f1, y_true, y_pred


def fedavg(models_weights):
    """Simple average."""
    return {k: sum(m[k] for m in models_weights) / len(models_weights) for k in models_weights[0].keys()}


def fedavg_weighted(models_weights, sizes):
    """Weighted FedAvg by dataset size."""
    total = sum(sizes)
    return {k: sum(m[k] * (s / total) for m, s in zip(models_weights, sizes)) for k in models_weights[0].keys()}


# ------------------------------------------------
# Federated Learning Trainer Class
# ------------------------------------------------
class FedAvgTrainer:
    def __init__(self, input_dim, output_dim,
                 n_clients=5, hidden_dim=64,
                 local_epochs=5, rounds=10,
                 lr=1e-3, weighted=False,
                 device=None):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_clients = n_clients
        self.hidden_dim = hidden_dim
        self.local_epochs = local_epochs
        self.rounds = rounds
        self.lr = lr
        self.weighted = weighted
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.global_model = MLP(input_dim, output_dim, hidden_dim).to(self.device)
        self.history = {'loss': [], 'acc': [], 'f1': []}

    # ------------------------------------------------
    # Prepare federated datasets (split by proportion)
    # ------------------------------------------------
    def prepare_clients(self, X_train, y_train, X_test, y_test, batch_size=32):
        dataset_size = len(X_train)
        proportions = [0.4, 0.25, 0.15, 0.1, 0.1][:self.n_clients]
        split_sizes = [int(p * dataset_size) for p in proportions]
        split_sizes[-1] += dataset_size - sum(split_sizes)
        indices = torch.randperm(dataset_size)
        X_train, y_train = X_train[indices], y_train[indices]
        X_splits = torch.split(X_train, split_sizes)
        y_splits = torch.split(y_train, split_sizes)

        print("\n📊 Client data distribution:")
        for i, s in enumerate(split_sizes, 1):
            print(f"Client {i}: {s} samples ({s / dataset_size:.1%})")

        self.client_loaders = [
            DataLoader(TensorDataset(X_splits[i], y_splits[i]), batch_size=batch_size, shuffle=True)
            for i in range(self.n_clients)
        ]
        self.client_sizes = split_sizes
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # ------------------------------------------------
    # Run full FL process
    # ------------------------------------------------
    def run(self):
        print("\n🚀 Starting Federated Learning")
        criterion = nn.CrossEntropyLoss()

        for r in range(1, self.rounds + 1):
            print(f"\n🌍 Round {r}/{self.rounds}")
            local_weights = []

            for i, loader in enumerate(self.client_loaders):
                model_local = copy.deepcopy(self.global_model)
                opt = torch.optim.Adam(model_local.parameters(), lr=self.lr)
                local_loss = train_local(model_local, loader, opt, criterion, self.device, self.local_epochs)
                local_weights.append(copy.deepcopy(model_local.state_dict()))
                print(f"   Client {i+1}: local loss = {local_loss:.4f}")

            if self.weighted:
                new_state = fedavg_weighted(local_weights, self.client_sizes)
            else:
                new_state = fedavg(local_weights)

            self.global_model.load_state_dict(new_state)

            test_loss, test_acc, test_f1, _, _ = evaluate(self.global_model, self.test_loader, criterion, self.device)
            self.history['loss'].append(test_loss)
            self.history['acc'].append(test_acc)
            self.history['f1'].append(test_f1)
            print(f"   Global model - Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

        print("\n✅ Federated Training Complete!")
        return self.global_model

    def plot_metrics(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], label='Loss')
        plt.plot(self.history['acc'], label='Accuracy')
        plt.plot(self.history['f1'], label='F1 Score')
        plt.xlabel('Round')
        plt.ylabel('Metric Value')
        plt.title('Federated Learning Performance per Round')
        plt.legend()
        plt.grid(True)
        plt.show()


# ------------------------------------------------
# Example usage
# ------------------------------------------------
if __name__ == "__main__":
    # Example with dummy data (replace this with your merged dataset)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=30, n_classes=2, random_state=42)
    X_train, X_test = X[:1600], X[1600:]
    y_train, y_test = y[:1600], y[1600:]

    # Scale
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    trainer = FedAvgTrainer(input_dim=30, output_dim=2, n_clients=5, weighted=True)
    trainer.prepare_clients(X_train, y_train, X_test, y_test)
    trainer.run()
    trainer.plot_metrics()
