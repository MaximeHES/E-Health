"""
fl_plots.py — plots for Federated Learning results (eHealth)

This script:
- Re-runs federated training (unweighted & weighted) with fl_train.run_federated
- Plots accuracy/loss vs rounds
- Compares weighted vs unweighted
- Shows client data distribution (samples per CenterID)
- Saves PNGs and CSVs under ./Ressources/plots/

Usage:
    python fl_plots.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fl_train import run_federated
from fl_utils import load_and_merge, prepare_features, make_client_loaders


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_curve(y, title, ylabel, out_path):
    plt.figure(figsize=(7,4))
    sns.lineplot(x=range(1, len(y)+1), y=y, marker="o")
    plt.xlabel("Rounds")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_dual(curves, labels, title, ylabel, out_path):
    plt.figure(figsize=(7,4))
    rounds = range(1, len(curves[0])+1)
    for y, lab in zip(curves, labels):
        sns.lineplot(x=rounds, y=y, marker="o", label=lab)
    plt.xlabel("Rounds")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_client_distribution(df_train, out_path):
    counts = df_train["CenterID"].value_counts().sort_index()
    plt.figure(figsize=(7,4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xlabel("Client (CenterID)")
    plt.ylabel("Num samples (train)")
    plt.title("Data distribution per client (train split)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_history_csv(history, out_csv):
    """Save history dict to CSV with columns: round, global_acc, global_loss"""
    rounds = len(history.get("global_acc", []))
    df = pd.DataFrame({
        "round": np.arange(1, rounds+1),
        "global_acc": history.get("global_acc", []),
        "global_loss": history.get("global_loss", []),
    })
    df.to_csv(out_csv, index=False)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    RES = os.path.join(base, "Ressources")
    PLOTS = os.path.join(RES, "plots")
    ensure_dir(PLOTS)

    # 1) Re-run FL to collect histories (simple & deterministic with fixed seed in fl_utils)
    print("\n[fl_plots] Running FL (unweighted) to collect history...")
    hist_u = run_federated(RES, local_epochs=1, rounds=10, batch_size=32, lr=1e-3, weighted=False)

    print("\n[fl_plots] Running FL (weighted) to collect history...")
    hist_w = run_federated(RES, local_epochs=1, rounds=10, batch_size=32, lr=5e-4, weighted=True)

    # 2) Plot curves
    plot_curve(hist_u["global_acc"], "Global Accuracy vs Rounds — Unweighted FedAvg",
               "Accuracy", os.path.join(PLOTS, "acc_unweighted.png"))
    plot_curve(hist_u["global_loss"], "Global Loss vs Rounds — Unweighted FedAvg",
               "Loss (BCEWithLogits)", os.path.join(PLOTS, "loss_unweighted.png"))

    plot_dual([hist_u["global_acc"], hist_w["global_acc"]],
              ["Unweighted", "Weighted"],
              "Accuracy vs Rounds — Unweighted vs Weighted FedAvg",
              "Accuracy", os.path.join(PLOTS, "acc_weighted_vs_unweighted.png"))

    plot_dual([hist_u["global_loss"], hist_w["global_loss"]],
              ["Unweighted", "Weighted"],
              "Loss vs Rounds — Unweighted vs Weighted FedAvg",
              "Loss (BCEWithLogits)", os.path.join(PLOTS, "loss_weighted_vs_unweighted.png"))

    # 3) Client data distribution (train split)
    df_train = load_and_merge(RES, split="train")
    plot_client_distribution(df_train, os.path.join(PLOTS, "client_distribution.png"))

    # 4) Save CSVs of histories
    save_history_csv(hist_u, os.path.join(PLOTS, "history_unweighted.csv"))
    save_history_csv(hist_w, os.path.join(PLOTS, "history_weighted.csv"))

    print("\n[fl_plots] Plots saved under:", PLOTS)
    print(" - acc_unweighted.png")
    print(" - loss_unweighted.png")
    print(" - acc_weighted_vs_unweighted.png")
    print(" - loss_weighted_vs_unweighted.png")
    print(" - client_distribution.png")
    print("Histories saved:")
    print(" - history_unweighted.csv")
    print(" - history_weighted.csv")


if __name__ == "__main__":
    main()
