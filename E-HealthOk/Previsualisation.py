#--------------------------------
#--------------------------------
# visualize and describe the data with plot
#--------------------------------
#--------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load data
df1_train = pd.read_csv('data_hn_clinical_train.csv')
df2_train = pd.read_csv('data_hn_ct_train.csv')
df3_train = pd.read_csv('data_hn_pt_train.csv')

df1_test = pd.read_csv('data_hn_clinical_test.csv')
df2_test = pd.read_csv('data_hn_ct_test.csv')
df3_test = pd.read_csv('data_hn_pt_test.csv')

dataset= df1_train.merge(df2_train, on=['PatientID', 'CenterID', 'Outcome'], how='inner') \
    .merge(df3_train, on=['PatientID', 'CenterID', 'Outcome'], how='inner')\



plt.figure(figsize=(12, 6))
colors = ['red', 'blue']

# Bar Chart
plt.subplot(1, 2, 1)
outcome_counts = dataset['Outcome'].replace({0: 'HPV-', 1: 'HPV+'}).value_counts()
outcome_counts.plot(kind='bar', color=colors, edgecolor='black')
for i, count in enumerate(outcome_counts):
    plt.text(i, count + 2, str(count), ha='center', fontsize=12)
plt.title('Outcome Distribution in the Train data')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks(range(len(outcome_counts)), outcome_counts.index)

#Pie Chart
plt.subplot(1, 2, 2)
outcome_counts.plot(kind='pie', colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
plt.ylabel('')
plt.title('Outcome Distribution')

plt.tight_layout()
plt.show()

#predicitive features
# Compute correlations
corr = dataset.corr(numeric_only=True)
top_corr = corr['Outcome'].drop('Outcome').sort_values(key=abs, ascending=False).head(10)

# Simple barplot
plt.figure(figsize=(7, 5))
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, palette='coolwarm', legend=False)

plt.title('Top 10 Features Most Correlated with Outcome')
plt.xlabel('Correlation with Outcome')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# data by center ID
#less relevant but still interesting for later (federated learning because of the repartition by center/hospital)
plt.figure(figsize=(12, 6))
colors = ['red', 'blue']
# Replace Outcome values consistently for the hospital plot
dataset['Outcome_Label'] = dataset['Outcome'].replace({0: 'HPV-', 1: 'HPV+'})
sns.countplot(data=dataset, x='CenterID', hue='Outcome_Label', edgecolor='black')
plt.title('Outcome by hospital (training set)')
plt.xlabel('Center ID')
plt.ylabel('Count')
plt.legend(title='Outcome', labels=['HPV positiv', 'HPV negativ'])
plt.xticks(rotation=45)

# Add counts on the bars
for bar in plt.gca().patches:
    bar_height = bar.get_height()
    if bar_height > 0:
        bar_width = bar.get_width()
        bar_x = bar.get_x()
        plt.text(bar_x + bar_width / 2, bar_height + 2, int(bar_height),
                 ha='center', va='bottom', fontsize=10, color='black')

plt.show()


# PCA plot
# Prepare data for PCA
if 'Gender' in dataset.columns:
    dataset['Gender'] = dataset['Gender'].replace({'M': 0, 'F': 1})

#numerical features and  missing values
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).drop(columns=['Outcome']).dropna()
outcome_labels = dataset.loc[numerical_features.index, 'Outcome']

#number of features used
num_features_used = numerical_features.shape[1]
print(f"Number of features used for PCA: {num_features_used}")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
pca_df['Outcome'] = outcome_labels.values
plt.figure(figsize=(10, 8))
point_count = len(pca_df)
outcome_0_count = (pca_df['Outcome'] == 0).sum()
outcome_1_count = (pca_df['Outcome'] == 1).sum()
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Outcome', edgecolor='black', s=100)
plt.title(f'PCA , {point_count} Points: {outcome_0_count} HPV-, {outcome_1_count} HPV+; {num_features_used} Features Used) out of {dataset.shape[1]-3} total')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Outcome', labels=['HPV+', 'HPV-'])
plt.show()