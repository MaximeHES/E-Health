# main.py
"""
Main script for the eHealth Machine Learning pipeline.
- Loads and cleans data (using clean.py)
- Trains a baseline model (using model_training.py)
- Evaluates and saves results
"""

import os
import pandas as pd
from model_training import ModelTraining

# -----------------------------
# 1. Define paths
# -----------------------------
BASE_PATH = os.path.join(".", "Ressources")

PATH_Clinical = os.path.join(BASE_PATH, "data_hn_clinical_test.csv")
PATH_CT       = os.path.join(BASE_PATH, "data_hn_ct_test.csv")
PATH_PT       = os.path.join(BASE_PATH, "data_hn_pt_test.csv")

# -----------------------------
# 2. Instantiate the trainer
# -----------------------------
trainer = ModelTraining(PATH_Clinical, PATH_CT, PATH_PT)

# -----------------------------
# 3. Load & clean all datasets
# -----------------------------
print("\n=== STEP 1: CLEANING DATASETS ===")
dfcli, dfct, dfpt = trainer.load_and_clean_data()

print("\nShapes after cleaning:")
print(f"Clinical: {dfcli.shape}")
print(f"CT: {dfct.shape}")
print(f"PT: {dfpt.shape}")

# -----------------------------
# 4. Merge datasets
# -----------------------------
print("\n=== STEP 2: MERGING DATASETS ===")
df_all = trainer.merge_datasets(dfcli, dfct, dfpt)
print(f"Merged dataset shape: {df_all.shape}")

# -----------------------------
# 5. Prepare data (split + scale)
# -----------------------------
print("\n=== STEP 3: PREPARING TRAIN/TEST DATA ===")
X_train, X_test, y_train, y_test = trainer.prepare_data(df_all)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# 6. Train baseline model (kNN)
# -----------------------------
print("\n=== STEP 4: TRAINING BASELINE MODEL (kNN) ===")
model = trainer.train_knn(X_train, y_train, n_neighbors=5)

# -----------------------------
# 7. Evaluate performance
# -----------------------------
print("\n=== STEP 5: EVALUATION ===")
metrics = trainer.evaluate(X_test, y_test)

# -----------------------------
# 8. (Optional) Save results
# -----------------------------
print("\n=== STEP 6: SAVING CLEANED FILES ===")
dfcli.to_csv(os.path.join(BASE_PATH, "data_hn_clinical_clean.csv"), index=False)
dfct.to_csv(os.path.join(BASE_PATH, "data_hn_ct_clean.csv"), index=False)
dfpt.to_csv(os.path.join(BASE_PATH, "data_hn_pt_clean.csv"), index=False)

print("Cleaned CSV files saved in 'Ressources' folder.")

# -----------------------------
# 9. Summary
# -----------------------------
print("\n=== PIPELINE SUMMARY ===")
print(f"Final merged dataset: {df_all.shape}")
print(f"Model performance: {metrics}")
print("\nPipeline completed successfully.")
