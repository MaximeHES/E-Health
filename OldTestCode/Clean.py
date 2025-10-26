import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional but makes plots prettier

# Load the dataset

PATH_Clinical = r".\Ressources\data_hn_clinical_test.csv"
PATH_CT = r".\Ressources\data_hn_ct_test.csv"
PATH_PT = r".\Ressources\data_hn_pt_test.csv"

dfcli = pd.read_csv(PATH_Clinical)
dfct = pd.read_csv(PATH_CT)
dfpt = pd.read_csv(PATH_PT)


#--- Functions for cleaning and checking data ---

#check for duplicates in each dataframe
def check_duplicates(df, name, subset=None):
    dup_count = df.duplicated(subset=subset).sum()
    print(f"\nDuplicate check for {name}")
    print("-" * 50)
    print(f"Number of duplicate rows: {dup_count}")
    if dup_count > 0:
        print("\nPreview of duplicate rows:")
        print(df[df.duplicated(keep=False)].head(10))
    df = df.drop_duplicates(subset=subset, keep="first")
    return df

#Structural errors fixer
def fix_structural_errors(df: pd.DataFrame, source: str = "clinical") -> pd.DataFrame:

    df = df.copy()

    #Clean column names
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    df.columns = pd.Index(df.columns).str.strip()

    #Source-specific cleaning
    if source.lower() == "clinical":
        # Text fields
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()
            df["Gender"].replace({"MALE": "M", "FEMALE": "F"}, inplace=True)

        if "CenterID" in df.columns:
            df["CenterID"] = df["CenterID"].astype(str).str.strip().str.title()

        # Binary columns
        binary_cols = ["Tobacco", "Alcohol", "Surgery", "Chemotherapy", "Outcome"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # keep NaN for now; imputation happens later if you want
                df[col] = df[col].clip(lower=0, upper=1)

        # Numeric columns
        num_cols = ["Age", "Weight", "Performance_status"]
        # Support both with/without underscore
        if "Performance status" in df.columns:
            df.rename(columns={"Performance status": "Performance_status"}, inplace=True)
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # IDs
        for id_col in ["PatientID", "CenterID"]:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).str.strip()

    elif source.lower() in {"ct", "pt"}:
        # IDs
        if "PatientID" in df.columns:
            df["PatientID"] = df["PatientID"].astype(str).str.strip()
        if "CenterID" in df.columns:
            df["CenterID"] = df["CenterID"].astype(str).str.strip().str.title()

        # Cast all non-ID columns to numeric
        non_id_cols = [c for c in df.columns if c not in {"PatientID", "CenterID"}]
        df[non_id_cols] = df[non_id_cols].apply(pd.to_numeric, errors="coerce")

    #Replace common textual missing tokens with NaN
    df.replace(
        to_replace=["?", "NA", "N/A", "na", "Nan", "nan", "NaN", "None", "none", "null", "Null", "-"],
        value=np.nan,
        inplace=True
    )

    #Drop fully empty columns
    df.dropna(axis=1, how="all", inplace=True)

    return df

# Missing values handler
def handle_missing_values(df: pd.DataFrame, source: str = "clinical") -> pd.DataFrame:
    """
    Handle missing values:
    - Fill categorical/binary NaN with 0
    - Fill continuous variables with median
    """
    df = df.copy()

    if source.lower() == "clinical":
        # Fill Tobacco & Alcohol with 0
        for col in ["Tobacco", "Alcohol"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        # Fill Performance status with median
        if "Performance_status" in df.columns:
            median_val = df["Performance_status"].median()
            df["Performance_status"].fillna(median_val, inplace=True)

        # Optional: verify remaining missing
        missing = df.isna().sum().sum()
        print(f"Remaining NaN values (clinical): {missing}")

    elif source.lower() in {"ct", "pt"}:
        # For radiomics data: fill NaN with column median (continuous features)
        df.fillna(df.median(numeric_only=True), inplace=True)
        print(f"Remaining NaN values ({source}): {df.isna().sum().sum()}")

    return df




#--- Appeling the functions ---
# 1) structural fixes
dfcli = fix_structural_errors(dfcli, source="clinical")
dfct  = fix_structural_errors(dfct,  source="ct")
dfpt  = fix_structural_errors(dfpt,  source="pt")

# 2) duplicates (after structural normalization)
dfcli = check_duplicates(dfcli, "Clinical Data", subset=["PatientID"])
dfct  = check_duplicates(dfct,  "CT Data",       subset=["PatientID"])
dfpt  = check_duplicates(dfpt,  "PT Data",       subset=["PatientID"])

# 3) missing values
dfcli = handle_missing_values(dfcli, source="clinical")
dfct  = handle_missing_values(dfct,  source="ct")
dfpt  = handle_missing_values(dfpt,  source="pt")



# final sanity checks
dfcli = check_duplicates(dfcli, "Clinical Data after cleaning", subset=["PatientID"])
dfct  = check_duplicates(dfct,  "CT Data after cleaning",       subset=["PatientID"])
dfpt  = check_duplicates(dfpt,  "PT Data after cleaning",       subset=["PatientID"])

print("\nShapes after cleaning:")
print(f"Clinical: {dfcli.shape} | CT: {dfct.shape} | PT: {dfpt.shape}")

