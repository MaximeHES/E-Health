#--------------------------------
#--------------------------------
# visualize and describe the data
#https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning
#https://dev.to/foyzulkarim/understanding-machine-learning-model-types-a-practical-guide-294e
#https://towardsdatascience.com/a-comprehensive-guide-to-multilayer-perceptrons-mlp-using-pytorch-5f6f6f2c22b3
#https://www.youtube.com/watch?v=aircAruvnKk
#https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
#--------------------------------
#--------------------------------

import seaborn as sns
from matplotlib import pyplot as plt
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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
#--------------------------------
# random forest (for comparaison)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib





# data load
df1_train = pd.read_csv('data_hn_clinical_train.csv')
df2_train = pd.read_csv('data_hn_ct_train.csv')
df3_train = pd.read_csv('data_hn_pt_train.csv')

df1_test = pd.read_csv('data_hn_clinical_test.csv')
df2_test = pd.read_csv('data_hn_ct_test.csv')
df3_test = pd.read_csv('data_hn_pt_test.csv')


#more sense to merge directly so we can see all features together since its for 1 patient.
# Merge csv train files
merged_train_df = df1_train.merge(df2_train, on=['PatientID', 'CenterID', 'Outcome'], how='inner') \
    .merge(df3_train, on=['PatientID', 'CenterID', 'Outcome'], how='inner')

# Merge csv test files
merged_test_df = df1_test.merge(df2_test, on=['PatientID', 'CenterID', 'Outcome'], how='inner') \
    .merge(df3_test, on=['PatientID', 'CenterID', 'Outcome'], how='inner')

#--------------------------------
# take a look at the merged data
#--------------------------------
print(merged_train_df.describe())
print(merged_train_df.info())
print(merged_train_df.head())
print(merged_test_df.describe())
print(merged_test_df.info())
print(merged_test_df.head())

# a bit of number in the data
print("Training data shape:", merged_train_df.shape)
print("Test data shape:", merged_test_df.shape)

#count numerical and categorical columns
num_cols = merged_train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = merged_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}")
print ("Numerical columns:", num_cols)
print ("Categorical columns:", cat_cols)
print("\n")
#distrbution of the data to see initial balance
print("Training data Outcome :")
print(merged_train_df['Outcome'].value_counts())
print("Test data Outcome :")
print(merged_test_df['Outcome'].value_counts())
print("\n")

#--------------------------------
# data cleaning
# focus on train data
#--------------------------------
dup_count_train = merged_train_df.duplicated().sum()
print(f"Number of duplicate rows in training data: {dup_count_train}")

# Handling missing values
merged_train_df.info()
cols = merged_train_df.columns
missing_numeric_values = merged_train_df[num_cols].isnull().sum()
print("Missing values in numerical columns:")
print(missing_numeric_values[missing_numeric_values > 0])
# Find rows with missing values in any of the numerical columns
rows_with_missing_values = merged_train_df[merged_train_df[cols].isnull().any(axis=1)]
print("Rows with missing values:")
print(rows_with_missing_values)
# test data
merged_test_df.info()
cols = merged_test_df.columns
missing_numeric_values_test = merged_test_df[num_cols].isnull().sum()
print("Missing values in numerical columns (test data):")
print(missing_numeric_values_test[missing_numeric_values_test > 0])
# Find rows with missing values in any of the numerical columns
rows_with_missing_values_test = merged_test_df[merged_test_df[cols].isnull().any(axis=1)]
print("Rows with missing values (test data):")
print(rows_with_missing_values_test)
#globally
print('Missing values in each column(train):')
print(missing_numeric_values)
print('Missing values in each column(test):')
print(missing_numeric_values_test)

#Map gender to numerical values
#because we will use only numerical features for the model
if 'Gender' in merged_test_df.columns:
    merged_test_df['Gender'] = merged_test_df['Gender'].map({'M': 1, 'F': 0})
if 'Gender' in merged_train_df.columns:
    merged_train_df['Gender'] = merged_train_df['Gender'].map({'M': 1, 'F': 0})

#preprocess the data according to cleaning part
# use iterative imputer for missing values instead of just putting 0 for missing values to have better estimate
# putting 0 can bias the data
#putting median can also be an option but will put same value for all missing values...
#imputer is based on other features to estimate missing values
print ("\n""Imputing missing values...")
imputer = IterativeImputer(max_iter=10, random_state=0)
imputer.fit(merged_train_df[num_cols])
merged_train_df[num_cols] = imputer.transform(merged_train_df[num_cols])
imputer.fit(merged_test_df[num_cols])
merged_test_df[num_cols] = imputer.transform(merged_test_df[num_cols])
#check if any missing values left
print("Missing values after imputation (train):")
print(merged_train_df[num_cols].isnull().sum())
print("Missing values after imputation (test):")
print(merged_test_df[num_cols].isnull().sum())
#--------------------------------

# still need to force values to 0 or 1, because after imputation some values can be 0.45 for example
# because the imputer uses regression so it predicts continuous numbers between 0 and 1, not just discrete 0/1.
#to understand -> 0.3 means "probably non-smoker" || 0.7 means "probably smoker

#value of Tobacco is < 0,5, consider it as 0 (non-smoker)
merged_train_df['Tobacco'] = merged_train_df['Tobacco'].apply(lambda x: 0 if x < 0.5 else 1)
merged_test_df['Tobacco'] = merged_test_df['Tobacco'].apply(lambda x: 0 if x < 0.5 else 1)
#value of Alcohol is < 0,5, consider it as 0 (non-drinker)
merged_train_df['Alcohol'] = merged_train_df['Alcohol'].apply(lambda x: 0 if x < 0.5 else 1)
merged_test_df['Alcohol'] = merged_test_df['Alcohol'].apply(lambda x: 0 if x < 0.5 else 1)
#no need for performance status as its not a feature but a score
merged_test_df.head(20)

# export cleaned data
output_dir = "CleanedData"
os.makedirs(output_dir, exist_ok=True)

merged_train_path = os.path.join(output_dir, "cleaned_train.csv")
merged_test_path = os.path.join(output_dir, "cleaned_test.csv")

merged_train_df.to_csv(merged_train_path, index=False)
merged_test_df.to_csv(merged_test_path, index=False)

print(f"\nCleaned datasets exported:")
print(f"  → {merged_train_path}")
print(f"  → {merged_test_path}")


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# train model
#--------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------


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
# MLP Model
# -------------------------------


#  STEP 2 — Prepare data for training
def prepare_from_cleaned(merged_train_df, merged_test_df):
    # Remove IDs, centers..
    drop_cols = ['PatientID', 'CenterID']
    for col in drop_cols:
        if col in merged_train_df.columns:
            merged_train_df = merged_train_df.drop(columns=[col])
        if col in merged_test_df.columns:
            merged_test_df = merged_test_df.drop(columns=[col])

    # Split features and labels
    X_train = merged_train_df.drop(columns=['Outcome'])
    y_train = merged_train_df['Outcome'].astype(int)
    X_test = merged_test_df.drop(columns=['Outcome'])
    y_test = merged_test_df['Outcome'].astype(int)

    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)

    # DataLoaders
    BATCH_SIZE = 32
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_t.shape[1]
    output_dim = len(np.unique(y_train))
    return train_loader, val_loader, test_loader, input_dim, output_dim



#  STEP 3 — Define MLP model

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



#  STEP 4 — Training / Evaluation loops
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)



#  STEP 5 — Main Training Routine
def train_mlp_on_clean_data(merged_train_df, merged_test_df):
    train_loader, val_loader, test_loader, INPUT_DIM, OUTPUT_DIM = prepare_from_cleaned(merged_train_df, merged_test_df)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(INPUT_DIM, OUTPUT_DIM, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10
    EPOCHS = 100

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    # save the model
    model_path = "trained_mlp_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n saved to {model_path}")

    # confusion matrix & metrics
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            preds = model(features).argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"F1: {f1_score(y_true, y_pred):.3f}, Precision: {precision_score(y_true, y_pred):.3f}, Recall: {recall_score(y_true, y_pred):.3f}")
    #print support for each class
    from sklearn.metrics import classification_report
    print("\nClassification Report:\n", classification_report(y_true, y_pred))


#--------------------------------
#--------------------------------
# Cross-validation (extra)
# repeats training several times on different splits of your data, then averages the results.
#Fold 1: Train on patients 1–80, test on 81–100
#Fold 2: Train on patients 1–60 + 81–100, test on 61–80
#Fold 3: Train on patients 1–40 + 61–100, test on 41–60
#...
#--------------------------------
#--------------------------------
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

def cross_validate_mlp(merged_df, n_splits=5, epochs=30):

    print(f"\nRunning {n_splits}-Fold Cross-Validation...")
    drop_cols = ['Outcome', 'PatientID', 'CenterID']
    X = merged_df.drop(columns=[col for col in drop_cols if col in merged_df.columns])

    if 'Gender' in X.columns and X['Gender'].dtype == 'object':
        X['Gender'] = X['Gender'].map({'M': 1, 'F': 0})

    X = X.select_dtypes(include=[np.number])
    y = merged_df['Outcome'].astype(int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, f1_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # To tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

        # Model
        INPUT_DIM = X_train_t.shape[1]
        OUTPUT_DIM = len(np.unique(y))
        model = MLP(INPUT_DIM, OUTPUT_DIM, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train(model, train_loader, optimizer, criterion, device='cpu')

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(1).numpy()

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        acc_scores.append(acc)
        f1_scores.append(f1)
        print(f"Fold {fold + 1}: Accuracy={acc:.3f}, F1={f1:.3f}")

    print("\nMean CV Accuracy:", np.mean(acc_scores), "±", np.std(acc_scores))
    print("Mean CV F1:", np.mean(f1_scores), "±", np.std(f1_scores))



#  STEP 6 — Run Training
if __name__ == "__main__":
    # Option 1: Train on train/test data
    train_mlp_on_clean_data(merged_train_df, merged_test_df)

    # Option 2: Uncomment to run cross-validation on training data
    #cross validation is a method
    #cross_validate_mlp(merged_train_df, n_splits=5)


# ------------------------------------------------------------
#Random Forest just to compare
# ------------------------------------------------------------
print("\n--- Training Random Forest ---")

# Separate features and labels
X_train = merged_train_df.drop(['PatientID', 'CenterID', 'Outcome'], axis=1)
y_train = merged_train_df['Outcome'].astype(int)
X_test = merged_test_df.drop(['PatientID', 'CenterID', 'Outcome'], axis=1)
y_test = merged_test_df['Outcome'].astype(int)

# Define base model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, 30, None]
}

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Mean CV score (training): {grid_search.best_score_:.4f}")

# Evaluate on test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("\n--- Test Set Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------
importances = best_rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\nTop 10 Most Important Features:")
for i in sorted_idx[:10]:
    print(f"{X_train.columns[i]}: {importances[i]:.4f}")

# Plot top 15 features
top_n = 15
top_indices = sorted_idx[:top_n]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[top_indices][::-1], color='skyblue')
plt.yticks(range(top_n), X_train.columns[top_indices][::-1])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.tight_layout()
plt.show()