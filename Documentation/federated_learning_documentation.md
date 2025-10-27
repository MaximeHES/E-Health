# ⚙️ Federated Learning Documentation – eHealth Project

This document describes the structure, logic, and internal mechanisms of the **Federated Learning (FL)** implementation used in the eHealth project.

The system is built around two main modules:

- `fl_train.py`
- `fl_utils.py`

Both are implemented in **PyTorch**, using the **FedAvg** and **Weighted FedAvg** strategies for decentralized training.

---

## 🧠 Overall Architecture

```
                 +---------------------+
                 |  eHealth Dataset     |
                 +----------+-----------+
                            |
              +-------------v--------------+
              |     Data preprocessing     |
              |   (scaling, encoding, etc.)|
              +-------------+--------------+
                            |
              +-------------v--------------+
              |      Federated Training    |
              |     (fl_train.py main)     |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |   Utility functions (FL)   |
              |       (fl_utils.py)        |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |   Visualization (fl_plots) |
              +-----------------------------+
```

---

## 🧩 `fl_train.py` – Federated Training Engine

### 🎯 Purpose
`fl_train.py` manages the **Federated Learning process** across multiple simulated clients (e.g., hospital centers), each training a local model on its subset of the data.

The global model is updated through **Federated Averaging (FedAvg)**, either:
- **Unweighted**: All clients contribute equally.
- **Weighted**: Clients contribute proportionally to their dataset size.

---

### 🧱 Key Components

#### 1. `get_weights(model)`
Returns a Python dictionary containing a copy of the model’s parameters.  
Used to exchange model weights between clients and the server.

#### 2. `set_weights(model, weights)`
Applies a given dictionary of weights to a PyTorch model.

#### 3. `fedavg_unweighted(weights_list)`
Performs **simple averaging** of model weights:
```python
avg[k] = sum(w[k] for w in weights_list) / len(weights_list)
```

#### 4. `fedavg_weighted(weights_list, sizes)`
Performs **weighted averaging** based on client dataset sizes:
```python
avg[k] = sum(w[k] * (size/total_size) for w, size in zip(weights_list, sizes))
```

#### 5. `local_train_epoch(model, loader, optimizer, device)`
Executes one **local training epoch** for a given client using **Binary Cross-Entropy with Logits Loss**.

Each client trains independently for a few epochs before sending updates to the server.

#### 6. `run_federated(...)`
The **core loop** of Federated Learning:
1. Load & merge datasets (`fl_utils.load_and_merge()`).
2. Prepare data (scaling, encoding).
3. Split by client (`CenterID`).
4. Initialize a global model (`MLPBinary`).
5. For each round:
   - Send global weights to clients.
   - Train local models.
   - Aggregate weights (FedAvg or Weighted FedAvg).
   - Evaluate global model on test set.
   - Apply **early stopping** if no improvement after `patience` rounds.

Returns a `history` dictionary:
```python
{
  "global_acc": [...],
  "global_loss": [...]
}
```

---

### 🧩 Example Workflow

```python
from Project.fl_train import run_federated

RES_PATH = "./Ressources"

# Run standard (unweighted) FedAvg
hist_u = run_federated(RES_PATH, rounds=10, weighted=False)

# Run weighted FedAvg
hist_w = run_federated(RES_PATH, rounds=10, weighted=True)
```

Output:
```
Clients: ['Center-3', 'Center-4', 'Center-5'] | sizes: [149, 189, 829]
Start FL: rounds=10, local_epochs=1, weighted=False
[Round 01] Test loss=0.6690 | acc=0.6096
...
Early stopping at round 9 (no improvement 3 rounds).
Final Global — loss=0.4999 | acc=0.7500
```

---

## ⚙️ `fl_utils.py` – Supporting Utilities

### 🎯 Purpose
This module contains helper functions for data preparation, model definition, and global evaluation used by `fl_train.py`.

---

### 🧱 Main Components

#### 1. `DEVICE` and `SEED`
- `DEVICE` automatically selects `"cuda"` if available, else `"cpu"`.
- `SEED` ensures deterministic training.

#### 2. `MLPBinary`
A **Multi-Layer Perceptron (MLP)** for binary classification:
```python
nn.Linear(input_dim, 128) → ReLU
nn.Linear(128, 64) → ReLU
nn.Linear(64, 1)
```
Used by both local clients and the global aggregator.

#### 3. `load_and_merge(resources_dir, split)`
Loads clinical, CT, and PET CSVs (train/test) and merges them on:
```python
["PatientID", "CenterID", "Outcome"]
```

#### 4. `prepare_features(df)`
Cleans and scales the dataset:
- Drops ID columns.
- Encodes categorical variables (e.g., Gender → 0/1).
- Standardizes numeric features using `StandardScaler`.

Returns:
```python
(X_df, X_scaled, y, scaler, columns)
```

#### 5. `make_loader(X, y, batch_size, shuffle)`
Creates a PyTorch `DataLoader` for mini-batch training.

#### 6. `make_client_loaders(df_train, scaler, columns, batch_size)`
Splits the training data by client (`CenterID`) and returns:
- A dict of DataLoaders per client.
- The dataset size of each client.

#### 7. `evaluate_binary(model, test_loader, device)`
Evaluates model performance (loss + accuracy) on test data using:
```python
criterion = nn.BCEWithLogitsLoss()
```
Applies a sigmoid activation to obtain probabilities.

---

### 🧠 Summary of Data Flow

```
Raw CSVs → load_and_merge() → prepare_features()
      ↓                           ↓
  Split by CenterID         Global scaling
      ↓                           ↓
make_client_loaders()       make_loader()
      ↓                           ↓
  Local client models        Test loader
      ↓                           ↓
   FedAvg aggregation     evaluate_binary()
```

---

## ✅ Key Features of the Implementation

| Feature | Description |
|----------|-------------|
| **Modular design** | Each step (data prep, training, aggregation) is isolated and reusable. |
| **Reproducibility** | Fixed random seed and deterministic behavior. |
| **Early stopping** | Stops FL when loss stops improving. |
| **Weighted aggregation** | Allows fair or biased influence based on data size. |
| **Plot-ready output** | Histories compatible with `fl_plots.py`. |
| **No saved models** | Keeps workspace clean; avoids .pt clutter. |

---

## 📊 Practical Notes

- Local models are **not persisted** (memory only).
- The global model weights are **recomputed at each round**.
- Histories are **returned as Python dictionaries**, exportable to CSV for plotting.
- The loss function is **BCEWithLogitsLoss**, ensuring numerical stability.

---

📘 *Prepared by Johann von Roten – eHealth Federated Learning Project*
