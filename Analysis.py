import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional but makes plots prettier

# Load the dataset

PATH_Clinical = r"C:\Users\Maxime\PycharmProjects\EHealthProject\Ressources\data_hn_clinical_test.csv"
PATH_CT = r"C:\Users\Maxime\PycharmProjects\EHealthProject\Ressources\data_hn_ct_test.csv"
PATH_PT = r"C:\Users\Maxime\PycharmProjects\EHealthProject\Ressources\data_hn_pt_test.csv"

df = pd.read_csv(PATH_Clinical)
# df = pd.read_csv(PATH_CT)
# df = pd.read_csv(PATH_PT)

# show full table info
pd.set_option('display.max_rows', None)  # show all rows
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  # don't wrap columns
pd.set_option('display.max_colwidth', None)  # show full column text

# full table
print(df)

# check what is duplicated
dup_count = df.duplicated().sum()
print("\n(b) Duplicate check")
print("-" * 50)
print(f"Number of duplicate rows: {dup_count}")
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
print("\nColumns with missing values in those rows:")
print(missing_rows.isna())


# Stats
# Check for outliers in numeric columns using IQR method
# Function to calculate lower and upper bounds

def calculate_iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


print(
    "\n(d) Outlier detection (IQR method)"
)
print("-" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    lower, upper = calculate_iqr_bounds(df[col])
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Column '{col}': {len(outliers)} outliers")
    if len(outliers) > 0:
        print(outliers[[col]].head())  # Show first few outlier values
# Optionally, visualize distributions with boxplots
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()
    plt.figure(figsize=(8, 4))
    sns.histplot(np.log1p(df[col]), kde=True)  # log1p
    plt.title(f'Log-Transformed Histogram of {col}')
    plt.show()
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=range(len(df)), y=df[col])
    plt.title(f'Scatter Plot of {col}')
    plt.show()
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=range(len(df)), y=np.log1p(df[col]))
    plt.title(f'Log-Transformed Scatter Plot of {col}')
    plt.show()




