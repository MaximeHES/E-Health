# 🧠 E-Health Project — Full PyTorch Pipeline Explanation

This document explains, step-by-step, how the script `ehealth_main_torch.py` works.  
It describes every part of the pipeline, why it was implemented that way, and how you can extend it later.

---

## 1️⃣ Imports and Initialization

The script starts by importing all required libraries:

- `pandas`, `numpy` for data manipulation.  
- `sklearn` for splitting, scaling, and metrics.  
- `torch`, `torch.nn`, `DataLoader` for the neural network.  

### Reproducibility
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
This ensures that every training run gives the same results (important for scientific or graded work).

---

## 2️⃣ Cleaning Functions

### `fix_structural_errors(df, source)`
- Standardizes column names (replaces spaces/hyphens with `_`).
- For **Clinical data**:
  - Cleans `Gender`, converts `"MALE"/"FEMALE"` to `"M"/"F"`.
  - Forces binary columns (`Tobacco`, `Alcohol`, `Surgery`, etc.) to numeric 0/1.
  - Renames `"Performance status"` to `"Performance_status"`.
  - Ensures numeric type for `Age`, `Weight`, and `Performance_status`.
- For **CT/PT data**:
  - Cleans identifiers.
  - Converts all other columns to numeric (radiomic features).
- Replaces invalid text (`"NA"`, `"?"`, `"None"`, etc.) with NaN.
- Drops fully empty columns.

### `handle_missing_values(df, source)`
- Clinical: fills missing values for `Tobacco` and `Alcohol` with 0, and `Performance_status` with median.  
- CT/PT: fills all missing values with the column median.

Result → clean numeric DataFrames, no NaNs.

---

## 3️⃣ Neural Network (MLP)

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```
- Two hidden layers: 128 → 64 → 1 neuron.  
- **ReLU** activations for non-linearity.  
- **Dropout 0.2** for regularization.  
- **Sigmoid** at the end → outputs a probability between 0 and 1.  

Loss function: **BCELoss**, optimizer: **Adam**.

---

## 4️⃣ Training Utilities

### `train_model()`
Handles one training epoch:
1. Puts model in training mode (`model.train()`).
2. For each batch:
   - Forward pass → compute prediction.
   - Compute loss.
   - Backward pass (`loss.backward()`).
   - Update weights (`optimizer.step()`).
3. Returns average epoch loss.

### `evaluate_model()`
Evaluates on test data:
- Disables gradient computation.
- Collects predictions (`Sigmoid` probabilities).
- Converts to 0/1 predictions (threshold = 0.5).
- Computes:
  - Accuracy
  - F1-score
  - ROC-AUC
  - Confusion matrix
  - Classification report (precision/recall per class)

---

## 5️⃣ Full Pipeline (main())

### a) File Paths
Uses paths relative to the script’s folder for robustness:
```python
base = os.path.dirname(os.path.abspath(__file__))
res  = os.path.join(base, "Ressources")
```

### b) Load and Clean
Reads the 3 datasets and applies the cleaning functions.

### c) Merge
```python
df = dfcli.merge(dfct, on=["PatientID","CenterID","Outcome"], how="inner")           .merge(dfpt, on=["PatientID","CenterID","Outcome"], how="inner")
```
Keeps only patients present in all three datasets.

### d) Encode Categorical Data
- `Gender` → 0/1 mapping.  
- `pd.get_dummies()` automatically encodes any remaining text columns.

### e) Features & Labels
```python
X = df.drop(columns=["PatientID","CenterID","Outcome"])
y = df["Outcome"]
```
Removes identifiers and the target column from the features.

### f) Train/Test Split and Scaling
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```
- **Stratified split** keeps class balance.  
- **StandardScaler**: fits only on the training set → avoids data leakage.

### g) DataLoaders
Converts data to PyTorch tensors and builds mini-batches for efficient training.

### h) Training Loop
Runs 30 epochs with loss printing every 5 epochs.

### i) Evaluation & Saving
After training:
- Computes metrics on the test set.
- Saves the model as `mlp_ehealth_model.pt` inside the `Ressources` folder.

---

## 6️⃣ Key Design Choices

| Concept | Implementation | Why |
|----------|----------------|----|
| Reproducibility | Seed fixed everywhere | Stable results |
| Feature scaling | StandardScaler | Normalizes radiomics features |
| Stratified split | `stratify=y` | Keeps balanced train/test classes |
| Regularization | Dropout + L2 | Prevents overfitting |
| Binary output | Sigmoid + BCELoss | For classification 0/1 |
| Evaluation | F1, ROC-AUC, Confusion Matrix | Interpretable and standard |
| GPU support | auto-detects CUDA | Faster training if available |

---

## 7️⃣ Summary of the Workflow

```
CSV → cleaning → merge → encoding → split (stratified) → scaling
    → tensors → dataloaders → MLP → training → evaluation → model.pt
```

---

## 8️⃣ Possible Improvements

- Add **Early Stopping**: monitor validation loss and stop when it no longer improves.  
- Add **Class Weights**: if classes are imbalanced.  
- Save `scaler` and `X.columns` for consistent inference later.  
- Add a small **validation set** (train/val/test split).  
- Visualize training curves (loss vs. epoch).

---

✅ **Conclusion**  
This script is a fully functional and well-structured baseline for binary outcome prediction in a multimodal medical dataset (Clinical + CT + PET).  
It follows professional ML standards: clean preprocessing, reproducibility, and interpretable evaluation.
