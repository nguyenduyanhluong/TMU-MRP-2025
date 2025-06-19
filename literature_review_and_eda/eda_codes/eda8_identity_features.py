# TMU MRP 2025
# EDA 8 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

transaction_df = pd.read_csv("cleaned_train_transaction.csv")
identity_df = pd.read_csv("cleaned_train_identity.csv")

df = pd.merge(transaction_df, identity_df, how='left', on='TransactionID')

# Figure 18: Distribution of id_02 by Fraud Class
plt.figure(figsize=(6, 4))
for label in [0, 1]:
    subset = df[df['isFraud'] == label]
    sns.kdeplot(subset['id_02'].dropna(), label=f'Fraud: {label}', fill=True)
plt.title("Distribution of id_02 by Fraud Class")
plt.xlabel("id_02")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Figure 19: Fraud Rate by Operating System (id_30)
top_os = df['id_30'].value_counts().head(10).index
os_fraud = df[df['id_30'].isin(top_os)].groupby('id_30')['isFraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(6, 4))
sns.barplot(x=os_fraud.values, y=os_fraud.index)
plt.xlabel("Fraud Rate")
plt.title("Fraud Rate by Operating System (id_30)")
plt.tight_layout()
plt.show()

# Figure 20: Fraud Rate by Browser (id_31)
top_browser = df['id_31'].value_counts().head(10).index
browser_fraud = df[df['id_31'].isin(top_browser)].groupby('id_31')['isFraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(6, 4))
sns.barplot(x=browser_fraud.values, y=browser_fraud.index)
plt.xlabel("Fraud Rate")
plt.title("Fraud Rate by Browser (id_31)")
plt.tight_layout()
plt.show()