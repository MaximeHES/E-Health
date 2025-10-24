# Data Cleaning Pipeline – eHealth Project

## Description
This script performs a complete cleaning process on three medical datasets used in the eHealth project:
- **Clinical data**: patient information (age, gender, habits, etc.)
- **CT data**: radiomics features extracted from CT scans (tumor structure)
- **PET data**: radiomics features extracted from PET scans (tumor metabolic activity)

The goal is to prepare consistent, standardized, and reliable data for machine learning and statistical analysis.

---

## 1. Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

- **pandas** – data loading, cleaning, and manipulation  
- **numpy** – numerical operations  
- **matplotlib / seaborn** – optional visualization libraries

---

## 2. Loading the Datasets

```python
PATH_Clinical = r".\Ressources\data_hn_clinical_test.csv"
PATH_CT       = r".\Ressources\data_hn_ct_test.csv"
PATH_PT       = r".\Ressources\data_hn_pt_test.csv"

dfcli = pd.read_csv(PATH_Clinical)
dfct  = pd.read_csv(PATH_CT)
dfpt  = pd.read_csv(PATH_PT)
```

Each dataset is loaded into a separate DataFrame.  
They are cleaned independently before being merged later on using `PatientID`.

---

## 3. Functions Overview

### 3.1 check_duplicates()
Removes duplicated rows from a dataset.

```python
def check_duplicates(df, name):
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate check for {name}")
    print(f"Number of duplicate rows: {dup_count}")
    if dup_count > 0:
        print(df[df.duplicated(keep=False)].head(10))
    df = df.drop_duplicates(subset=subset, keep="first")
    return df
```

**Purpose**
- Detects and removes duplicate rows.
- Uses `subset=["PatientID"]` to ensure each patient appears only once.

---

### 3.2 fix_structural_errors()
Cleans column names, data types, and formatting errors.

```python
def fix_structural_errors(df, source="clinical"):
```
**Actions performed**
| Step | Description |
|------|--------------|
| Column names | Replaces spaces and hyphens with underscores |
| Text normalization | Removes leading/trailing spaces, formats gender and center names |
| Type conversion | Converts `Age`, `Weight`, and `Performance_status` columns to numeric |
| Binary checks | Ensures binary variables stay between 0 and 1 |
| Missing tokens | Replaces `NA`, `None`, `?`, `-` etc. with `NaN` |
| Empty columns | Drops columns containing only NaN values |

---

### 3.3 handle_missing_values()
Handles missing data according to the data type.

```python
def handle_missing_values(df, source="clinical"):
```
**Strategies**
| Dataset | Column | Method | Reason |
|----------|---------|---------|--------|
| Clinical | `Tobacco`, `Alcohol` | Fill with 0 | Missing likely means "no" |
| Clinical | `Performance_status` | Fill with median | Keeps typical distribution |
| CT / PET | All numeric columns | Fill with column median | Standard imputation for radiomics |

---

### 3.4 detect_and_remove_outliers()
Detects and removes outliers using the Interquartile Range (IQR) method.

```python
def detect_and_remove_outliers(df, source="clinical"):
```
**Logic**
- Q1 = 25th percentile  
- Q3 = 75th percentile  
- IQR = Q3 - Q1  
- Values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` are considered outliers.  

Applied only to clinical columns: `Age`, `Weight`, and `Performance_status`.  
Outlier removal is skipped for CT and PET datasets to preserve radiomic integrity.

---

## 4. Pipeline Execution

```python
# 1. Structural fixes
dfcli = fix_structural_errors(dfcli, source="clinical")
dfct  = fix_structural_errors(dfct,  source="ct")
dfpt  = fix_structural_errors(dfpt,  source="pt")

# 2. Remove duplicates
dfcli = check_duplicates(dfcli, "Clinical Data", subset=["PatientID"])
dfct  = check_duplicates(dfct,  "CT Data",       subset=["PatientID"])
dfpt  = check_duplicates(dfpt,  "PT Data",       subset=["PatientID"])

# 3. Handle missing values
dfcli = handle_missing_values(dfcli, source="clinical")
dfct  = handle_missing_values(dfct,  source="ct")
dfpt  = handle_missing_values(dfpt,  source="pt")

# 4. Detect and remove outliers (clinical only)
dfcli = detect_and_remove_outliers(dfcli, source="clinical")
```

---

## 5. Final Checks

```python
dfcli = check_duplicates(dfcli, "Clinical Data after cleaning", subset=["PatientID"])
dfct  = check_duplicates(dfct,  "CT Data after cleaning",       subset=["PatientID"])
dfpt  = check_duplicates(dfpt,  "PT Data after cleaning",       subset=["PatientID"])

print("\nShapes after cleaning:")
print(f"Clinical: {dfcli.shape} | CT: {dfct.shape} | PT: {dfpt.shape}")
```

Displays remaining duplicates and final dimensions of each dataset.

---

## 6. Optional: Save Cleaned Files

```python
dfcli.to_csv(r".\Ressources\data_hn_clinical_clean.csv", index=False)
dfct.to_csv(r".\Ressources\data_hn_ct_clean.csv", index=False)
dfpt.to_csv(r".\Ressources\data_hn_pt_clean.csv", index=False)
```

---

## 7. Summary

| Step | Function | Description |
|------|-----------|-------------|
| 1 | `fix_structural_errors()` | Standardizes column names, types, and values |
| 2 | `check_duplicates()` | Removes duplicate entries |
| 3 | `handle_missing_values()` | Imputes missing values logically |
| 4 | `detect_and_remove_outliers()` | Removes implausible values from clinical data |
| 5 | `print shapes` | Confirms final dataset size |

---

## 8. Notes and Recommendations

- Always keep raw files untouched; save cleaned versions separately.  
- For CT and PET data, avoid deleting rows—apply robust scaling instead.  
- All cleaning operations occur in memory. To persist results, explicitly use `to_csv()`.  
- Document the number of rows removed and NaN values replaced for traceability.

---

## 9. Example Output

```
Duplicate check for Clinical Data
----------------------------------
Number of duplicate rows: 0

Remaining NaN values (clinical): 0
Age: keep within [41.50, 79.50]
Weight: keep within [50.25, 102.75]
Performance_status: keep within [0.00, 2.00]
Removed 3 rows as outliers (CLINICAL).

Shapes after cleaning:
Clinical: (57, 11) | CT: (57, 160) | PT: (57, 160)
```
