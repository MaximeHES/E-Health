"""
fl_train.py — simple, functional Federated Learning (FedAvg) for eHealth.
- Uses train split for clients (by CenterID)
- Evaluates on test split
- Supports unweighted and weighted FedAvg
"""

import os
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from fl_utils import (
    DEVICE, SEED,
    load_and_merge, prepare_features, align_and_scale,
    make_loader, make_client_loaders,
    MLPBinary, evaluate_binary
)


def get_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(weights)

def fedavg_unweighted(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg = {}
    for k in weights_list[0].keys():
        avg[k] = sum(w[k] for w in weights_list) / float(len(weights_list))
    return avg

def fedavg_weighted(weights_list: List[Dict[str, torch.Tensor]], sizes: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(sizes))
    avg = {}
    for k in weights_list[0].keys():
        s = None
        for w, sz in zip(weights_list, sizes):
            contrib = w[k] * (sz / total)
            s = contrib if s is None else s + contrib
        avg[k] = s
    return avg

def local_train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str = DEVICE) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits  = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_n    += xb.size(0)
    return total_loss / max(total_n,1)


def run_federated(resources_dir: str,
                  local_epochs: int = 1,
                  rounds: int = 10,
                  batch_size: int = 32,
                  lr: float = 1e-3,
                  weighted: bool = False):
    # 1) Load & merge
    df_train = load_and_merge(resources_dir, split="train")
    df_test  = load_and_merge(resources_dir, split="test")

    # 2) Prepare global features (fit scaler on global train)
    X_train_df, X_train_s, y_train, scaler, columns = prepare_features(df_train)
    X_test_df,  X_test_s,  y_test,  _,      _       = prepare_features(df_test)

    # Global test loader
    test_loader = make_loader(X_test_s, y_test, batch_size=batch_size, shuffle=False)

    # 3) Make client loaders per CenterID (aligned to global columns)
    loaders, sizes_dict = make_client_loaders(df_train, scaler, columns, batch_size=batch_size)
    client_ids = list(loaders.keys())
    sizes = [sizes_dict[cid] for cid in client_ids]

    # 4) Init global model
    input_dim = X_train_s.shape[1]
    model_global = MLPBinary(input_dim).to(DEVICE)

    print(f"Clients: {client_ids} | sizes: {sizes}")
    print(f"Start FL: rounds={rounds}, local_epochs={local_epochs}, weighted={weighted}")

    # 5) Federated rounds
    history = {"global_acc": [], "global_loss": []}
    best_loss = float("inf")
    patience = 3
    wait = 0
    for r in range(1, rounds+1):
        local_weights, local_sizes = [], []
        # train each client locally
        for cid in client_ids:
            loader = loaders[cid]
            model_local = MLPBinary(input_dim).to(DEVICE)
            set_weights(model_local, get_weights(model_global))
            opt = optim.Adam(model_local.parameters(), lr=lr)

            for _ in range(local_epochs):
                _ = local_train_epoch(model_local, loader, opt, DEVICE)

            local_weights.append(get_weights(model_local))
            local_sizes.append(sizes_dict[cid])

        # aggregate
        if weighted:
            new_w = fedavg_weighted(local_weights, local_sizes)
        else:
            new_w = fedavg_unweighted(local_weights)
        set_weights(model_global, new_w)

        # eval on global test
        loss_g, acc_g = evaluate_binary(model_global, test_loader, DEVICE)
        history["global_loss"].append(loss_g)
        history["global_acc"].append(acc_g)
        print(f"[Round {r:02d}] Test loss={loss_g:.4f} | acc={acc_g:.4f}")

        # --- early stopping on global loss ---
        if loss_g + 1e-6 < best_loss:
            best_loss = loss_g
            best_weights = get_weights(model_global)  # garde le meilleur modèle
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at round {r} (no improvement {patience} rounds).")
                set_weights(model_global, best_weights)  # restaure le meilleur
                break


    # Final evaluation
    loss_final, acc_final = evaluate_binary(model_global, test_loader, DEVICE)
    print(f"\nFinal Global — loss={loss_final:.4f} | acc={acc_final:.4f}")

    # Save model
    # Save then remove model (temporary file only)
    out_path = os.path.join(resources_dir, f"fl_mlp_binary_{'weighted' if weighted else 'unweighted'}.pt")
    torch.save(model_global.state_dict(), out_path)
    print(f"Saved global model temporarily to: {out_path}")

    # Immediately delete to keep workspace clean
    try:
        os.remove(out_path)
        print(f"Deleted temporary model file: {out_path}")
    except Exception as e:
        print(f"Could not delete {out_path}: {e}")

    return history


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    RES = os.path.join(base, "Ressources")

    # Run both variants for convenience
    print("\n=== Federated (Unweighted FedAvg) ===")
    hist_u = run_federated(RES, local_epochs=1, rounds=10, batch_size=32, lr=1e-3, weighted=False)

    print("\n=== Federated (Weighted FedAvg) ===")
    hist_w = run_federated(RES, local_epochs=1, rounds=10, batch_size=32, lr=1e-3, weighted=True)

    print("\nUnweighted acc by round:", [round(a,4) for a in hist_u['global_acc']])
    print("Weighted   acc by round:", [round(a,4) for a in hist_w['global_acc']])
