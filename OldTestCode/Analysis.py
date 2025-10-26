import Clean as Clean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



PATH_Clinical = r".\Ressources\data_hn_clinical_test.csv"
PATH_CT = r".\Ressources\data_hn_ct_test.csv"
PATH_PT = r".\Ressources\data_hn_pt_test.csv"

dfcli = pd.read_csv(PATH_Clinical)
dfct = pd.read_csv(PATH_CT)
dfpt = pd.read_csv(PATH_PT)


#clean using Clean.py functions

# 1) structural fixes
dfcli = Clean.fix_structural_errors(dfcli, source="clinical")
dfct  = Clean.fix_structural_errors(dfct, source="ct")
dfpt  = Clean.fix_structural_errors(dfpt, source="pt")

# 2) duplicates (after structural normalization)
dfcli = Clean.check_duplicates(dfcli, "Clinical Data", subset=["PatientID"])
dfct  = Clean.check_duplicates(dfct, "CT Data", subset=["PatientID"])
dfpt  = Clean.check_duplicates(dfpt, "PT Data", subset=["PatientID"])

# 3) missing values
dfcli = Clean.handle_missing_values(dfcli, source="clinical")
dfct  = Clean.handle_missing_values(dfct, source="ct")
dfpt  = Clean.handle_missing_values(dfpt, source="pt")

# 4) outliers (clinical only)
dfcli = Clean.detect_and_remove_outliers(dfcli, source="clinical")

# final sanity checks
dfcli = Clean.check_duplicates(dfcli, "Clinical Data after cleaning", subset=["PatientID"])
dfct  = Clean.check_duplicates(dfct, "CT Data after cleaning", subset=["PatientID"])
dfpt  = Clean.check_duplicates(dfpt, "PT Data after cleaning", subset=["PatientID"])

print("\nShapes after cleaning:")
print(f"Clinical: {dfcli.shape} | CT: {dfct.shape} | PT: {dfpt.shape}")


def perform_pca_plot(df, title, hue_col='Outcome'):
    """Perform PCA on numeric feature columns only (excluding the label),
    then plot 2D scatter colored by Outcome (if available)."""

    # 1️⃣ Select only numeric columns
    numeric_df = df.select_dtypes(include=['number']).copy()

    # 2️⃣ Remove Outcome if present (don’t include it in PCA)
    if hue_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[hue_col])

    # 3️⃣ Fill missing values
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    # 4️⃣ Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # 5️⃣ Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # 6️⃣ Create dataframe for visualization
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    # Add Outcome column back for color-coding (optional)
    if hue_col in df.columns:
        pca_df[hue_col] = df[hue_col].values

    # 7️⃣ Print explained variance info
    print(f"\n{title} PCA - Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance captured by first 2 PCs: {sum(pca.explained_variance_ratio_):.2f}")

    # 8️⃣ Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue=hue_col, data=pca_df, palette='Set2', s=60)
    plt.title(f"{title} - PCA (2D projection)")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title=hue_col)
    plt.show()


# Perform PCA and plot for each dataset
perform_pca_plot(dfcli, title="Clinical Data", hue_col='Outcome')
perform_pca_plot(dfct, title="CT Data", hue_col='Outcome')
perform_pca_plot(dfpt, title="PT Data", hue_col='Outcome')


# -----------------------------
# PCA on merged datasets
# -----------------------------

# Merge Clinical + CT + PT
df_all = (
    dfcli.merge(dfct, on="PatientID", how="inner")
         .merge(dfpt, on="PatientID", how="inner")
)

if "Outcome" not in df_all.columns and "Outcome" in dfcli.columns:
    df_all["Outcome"] = dfcli.set_index("PatientID").loc[df_all["PatientID"], "Outcome"].values

perform_pca_plot(df_all, "Clinical + CT + PT (All Features)")


# Merge Clinical + CT
df_cli_ct = dfcli.merge(dfct, on="PatientID", how="inner")

# Reattach Outcome only for visualization (not for PCA input)
if "Outcome" not in df_cli_ct.columns and "Outcome" in dfcli.columns:
    df_cli_ct["Outcome"] = dfcli.set_index("PatientID").loc[df_cli_ct["PatientID"], "Outcome"].values

# Run PCA (it will automatically exclude Outcome before analysis)
perform_pca_plot(df_cli_ct, "Clinical + CT (Subset Features)")


def plot_feature_contributions(df, title):
    numeric_df = df.select_dtypes(include=['number']).copy()
    if 'Outcome' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Outcome'])
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=2)
    pca.fit(scaled_data)

    # Get feature importance for PC1 & PC2
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=numeric_df.columns
    )

    top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10)
    top_features.plot(kind='barh', figsize=(8,5))
    plt.title(f"{title} - Top 10 Feature Contributions to PC1 & PC2")
    plt.xlabel("Absolute loading strength")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.show()

plot_feature_contributions(df_all, "Clinical + CT + PT")
