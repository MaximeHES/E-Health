import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional but makes plots prettier

# Load the dataset

PATH = r"C:\Users\Maxime\PycharmProjects\PandaTuto\sources\iris_to_clean.csv"
df = pd.read_csv(PATH)

print("a) Loaded dataset")
print("-" * 60)
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nBasic stats (numeric):")
print(df.describe())

# Check for duplicates
dup_count = df.duplicated().sum()
print("\n(b) Duplicate check")
print("-" * 50)
print(f"Number of duplicate rows: {dup_count}")

# check what is duplicated
if dup_count > 0:
    print("\nPreview of duplicate rows:")
    print(df[df.duplicated(keep=False)].head(10))


# Check for missing values

print("\n(c) Missing values check")
print("-" * 50)

# Count missing values in each column
missing_counts = df.isna().sum()

# Calculate percentage per column
missing_percent = (missing_counts / len(df) * 100).round(2)

# Combine into table, sorted by most missing
missing_table = (
    pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": missing_percent
    })
    .sort_values("missing_count", ascending=False)
)
print(missing_table)

# Show rows with any missing values
rows_with_missing = df[df.isna().any(axis=True)]
print(f"\nRows with missing values: {len(rows_with_missing)}")
print(rows_with_missing)

# Find which columns are missing in each of those 3 rows
missing_rows = df[df.isna().any(axis=True)]

for index, row in missing_rows.iterrows():
    missing_in_row = row[row.isna()].index.tolist()
    print(f"Row {index} -> missing in columns: {missing_in_row}")


# Fill missing values in numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if df[col].isna().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Filled missing in {col} with median ({median_value})")

# Fill missing values in categorical columns with mode
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

for col in categorical_cols:
    if df[col].isna().any():
        mode_value = df[col].mode(dropna=True)
        if not mode_value.empty:
            fill_value = mode_value.iloc[0]
        else:
            fill_value = "Unknown"
        df[col] = df[col].fillna(fill_value)
        print(f"Filled missing in {col} with mode ({fill_value})")

# Verify no missing values remain
print("\nAfter filling, any missing values remaining?")
print(df.isna().any())

#Median is less sensitive to extreme values (outliers).
#Mode keeps the most common label for categorical variables.


# Check for outliers in numeric columns using IQR method
# Function to calculate lower and upper bounds
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

print("\n(d) Outlier detection (IQR method)")
print("-" * 50)

numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_indices = set()

for col in numeric_cols:
    lower, upper = iqr_bounds(df[col])
    mask = (df[col] < lower) | (df[col] > upper)
    outliers_count = mask.sum()
    print(f"{col}: bounds=({lower:.3f}, {upper:.3f}), outliers={outliers_count}")
    if outliers_count > 0:
        outlier_indices.update(df.index[mask])


print(f"\nTotal unique rows with outliers: {len(outlier_indices)}")

# Show some outlier rows
if outlier_indices:
    print("\nPreview of outlier rows:")
    print(df.loc[list(outlier_indices)].head(10))

#Option 1: Remove them (if wrong)
#df_clean = df.drop(index=outlier_indices).reset_index(drop=True)
#print(f"After removing outliers: {df_clean.shape[0]} rows remain")

#Option 2: Cap them (if valid but extreme)
for col in numeric_cols:
    lower, upper = iqr_bounds(df[col])
    df[col] = np.clip(df[col], lower, upper)


print("\nNew summary after cleaning:")
print(df.describe())


# plot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['sepal.width'])
plt.title('Boxplot - Sepal Width')
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.boxplot(df['sepal.width'].dropna(), vert=True)
plt.title('Boxplot – sepal.width (current df)')
plt.ylabel('sepal.width')
plt.tight_layout()
plt.show()
