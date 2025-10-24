# ModelTraining Class – eHealth Project

## Overview
The `ModelTraining` class is responsible for handling the **machine learning phase** of the eHealth project.  
It loads the cleaned datasets, merges them, prepares the features for training, and evaluates baseline models.

This class complements the `clean.py` module, which performs all data preprocessing and cleaning.

---

## 1. Purpose
The goal of this class is to automate the following workflow:

1. Load **Clinical**, **CT**, and **PET** datasets from CSV files.  
2. Clean each dataset using functions from `clean.py`.  
3. Merge them into a single DataFrame on `PatientID`, `CenterID`, and `Outcome`.  
4. Split data into features (`X`) and target (`y`).  
5. Standardize (scale) features.  
6. Train and evaluate a simple **k-Nearest Neighbors (kNN)** model.  

---

## 2. Key Imports

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from clean import (
    fix_structural_errors,
    check_duplicates,
    handle_missing_values,
    detect_and_remove_outliers,
)
```

### Purpose of each library
| Library | Usage |
|----------|--------|
| `pandas`, `numpy` | Load and manipulate data |
| `sklearn.model_selection` | Split into training and testing sets |
| `sklearn.preprocessing` | Standardize features |
| `sklearn.neighbors` | k-Nearest Neighbors algorithm |
| `sklearn.metrics` | Model evaluation |
| `clean.py` | Import custom cleaning utilities |

---

## 3. Class Initialization

```python
class ModelTraining:
    def __init__(self, path_clinical, path_ct, path_pt):
        self.path_clinical = path_clinical
        self.path_ct = path_ct
        self.path_pt = path_pt
        self.dfcli = self.dfct = self.dfpt = self.df_all = None
        self.scaler = None
        self.model = None
```

**Explanation:**
- Stores the file paths to each dataset.
- Initializes empty attributes for the three DataFrames, the merged dataset, the scaler, and the model.

---

## 4. Data Loading and Cleaning

```python
def load_and_clean_data(self):
    dfcli = pd.read_csv(self.path_clinical)
    dfct = pd.read_csv(self.path_ct)
    dfpt = pd.read_csv(self.path_pt)

    dfcli = fix_structural_errors(dfcli, "clinical")
    dfct  = fix_structural_errors(dfct,  "ct")
    dfpt  = fix_structural_errors(dfpt,  "pt")

    dfcli = check_duplicates(dfcli, "Clinical", subset=["PatientID"])
    dfct  = check_duplicates(dfct,  "CT",       subset=["PatientID"])
    dfpt  = check_duplicates(dfpt,  "PT",       subset=["PatientID"])

    dfcli = handle_missing_values(dfcli, "clinical")
    dfct  = handle_missing_values(dfct,  "ct")
    dfpt  = handle_missing_values(dfpt,  "pt")

    dfcli = detect_and_remove_outliers(dfcli, "clinical")
    return dfcli, dfct, dfpt
```

**What it does:**
- Loads the CSVs into Pandas DataFrames.
- Cleans each dataset using the imported functions.
- Applies outlier removal only on the clinical data.

---

## 5. Dataset Merging

```python
def merge_datasets(self, dfcli, dfct, dfpt):
    df = dfcli.merge(dfct, on=["PatientID", "CenterID", "Outcome"])
    df = df.merge(dfpt, on=["PatientID", "CenterID", "Outcome"])
    self.df_all = df
    return df
```

**Explanation:**
- Merges the three cleaned DataFrames into a single dataset.
- Ensures that each patient’s data (clinical + CT + PET) are combined.
- The join keys guarantee proper patient alignment.

---

## 6. Feature Preparation and Data Splitting

```python
def prepare_data(self, df_merged, test_size=0.2, random_state=42):
    X = df_merged.drop(columns=["PatientID", "CenterID", "Outcome"])
    y = df_merged["Outcome"].astype(int)
    self.scaler = StandardScaler()
    X_scaled = self.scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
```

**Explanation:**
- Removes identifier and target columns to create `X` (features) and `y` (target).  
- Standardizes all features using `StandardScaler` to normalize the radiomic feature ranges.  
- Splits data into 80% training and 20% testing sets, preserving class proportions with `stratify=y`.

---

## 7. Model Training (kNN)

```python
def train_knn(self, X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    self.model = knn
    return knn
```

**Explanation:**
- Trains a simple **k-Nearest Neighbors** classifier.
- `n_neighbors` defines how many neighbors are considered for classification.
- The trained model is stored in `self.model` for later evaluation.

---

## 8. Model Evaluation

```python
def evaluate(self, X_test, y_test):
    y_pred = self.model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    if hasattr(self.model, "predict_proba"):
        y_score = self.model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)

    print("Evaluation metrics:", metrics)
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    return metrics
```

**Explanation:**
- Predicts outcomes on the test set.
- Calculates accuracy, F1 score, and ROC AUC (if probabilities are available).
- Prints a detailed classification report for precision, recall, and F1 per class.

---

## 9. Example Usage (main block)

```python
if __name__ == "__main__":
    PATH_Clinical = r".\Ressources\data_hn_clinical_test.csv"
    PATH_CT = r".\Ressources\data_hn_ct_test.csv"
    PATH_PT = r".\Ressources\data_hn_pt_test.csv"

    trainer = ModelTraining(PATH_Clinical, PATH_CT, PATH_PT)
    dfcli, dfct, dfpt = trainer.load_and_clean_data()
    df_all = trainer.merge_datasets(dfcli, dfct, dfpt)
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_all)
    trainer.train_knn(X_train, y_train, n_neighbors=5)
    trainer.evaluate(X_test, y_test)
```

**What happens when executed:**
1. Instantiates `ModelTraining` with the dataset paths.  
2. Loads and cleans the data.  
3. Merges all sources into one dataset.  
4. Prepares the data for training.  
5. Trains a kNN model.  
6. Evaluates and prints model performance.

---

## 10. Summary of the Workflow

| Step | Method | Description |
|------|---------|-------------|
| 1 | `load_and_clean_data()` | Loads and cleans the datasets |
| 2 | `merge_datasets()` | Merges all sources into one DataFrame |
| 3 | `prepare_data()` | Scales and splits data into train/test |
| 4 | `train_knn()` | Trains a baseline kNN classifier |
| 5 | `evaluate()` | Evaluates model performance |

---

## 11. Next Possible Enhancements

- Add more algorithms (Logistic Regression, Random Forest, MLP).  
- Implement cross-validation by `CenterID` to avoid data leakage.  
- Include model saving/loading (`joblib`).  
- Add feature importance visualization for interpretability.  

---
