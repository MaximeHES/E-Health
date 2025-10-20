import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Load data

TRAIN = r"C:\Users\Maxime\PycharmProjects\PandaTuto\sources\train_health_data.csv"
TEST  = r"C:\Users\Maxime\PycharmProjects\PandaTuto\sources\test_health_data.csv"

train_df = pd.read_csv(TRAIN)
test_df  = pd.read_csv(TEST)

TARGET = "Class"


# Feature and label columns

X_train_raw = train_df.drop(columns=[TARGET])
y_train_raw = train_df[TARGET].astype(str)

if TARGET in test_df.columns:
    X_test_raw  = test_df.drop(columns=[TARGET])
    y_test_raw  = test_df[TARGET].astype(str)
else:
    X_test_raw  = test_df.copy()
    y_test_raw  = None

# Keep only numeric features for this simple pipeline
num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
X_train_num = X_train_raw[num_cols].copy()
X_test_num  = X_test_raw[num_cols].copy()


# Convert labels to integers

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)          # e.g., {'At Risk':0, 'Disease':1, 'Healthy':2}
y_test  = le.transform(y_test_raw) if y_test_raw is not None else None


# 3) Handle missing and scale features

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp = imputer.fit_transform(X_train_num)
X_test_imp  = imputer.transform(X_test_num)

X_train = scaler.fit_transform(X_train_imp)
X_test  = scaler.transform(X_test_imp)


# 4) k-NN classifier

knn = KNeighborsClassifier(n_neighbors=7, weights="distance", p=2)


# 5) Fit the model
knn.fit(X_train, y_train)

# 6) Evaluate on the test set
y_pred = knn.predict(X_test)

if y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", round(acc, 4))

    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix with readable class names
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
else:
    print("Test file has no labels. Showing the first predictions:")
    print(pd.Series(le.inverse_transform(y_pred)).head())

# (Optional) Save predictions with original labels
out = X_test_raw.copy()
out["pred_Class"] = le.inverse_transform(y_pred)
if y_test_raw is not None:
    out["true_Class"] = y_test_raw.values
out.to_csv("health_test_predictions_simple.csv", index=False)
print("\nSaved predictions -> health_test_predictions_simple.csv")