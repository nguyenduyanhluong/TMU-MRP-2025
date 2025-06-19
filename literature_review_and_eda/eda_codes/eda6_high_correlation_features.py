# TMU MRP 2025
# EDA 6 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

transaction_df = pd.read_csv("cleaned_train_transaction.csv")

# Figure 12: Top 10 Features Correlated with isFraud
numeric_df = transaction_df.select_dtypes(include=['int64', 'float64'])

threshold = 0.5 
numeric_df = numeric_df.loc[:, numeric_df.isnull().mean() < (1 - threshold)]
numeric_df = numeric_df.drop(columns=["TransactionID"], errors="ignore")

correlation = numeric_df.corr(numeric_only=True)
fraud_corr = correlation["isFraud"].drop("isFraud").sort_values(key=abs, ascending=False)
top_corr = fraud_corr.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette="coolwarm")
plt.title("Top 10 Features Correlated with isFraud", fontsize=20)
plt.xlabel("Correlation with isFraud", fontsize=17)
plt.ylabel("V-series", fontsize=17)
plt.tight_layout()
plt.show()

print("Top 10 Correlated Features with isFraud:\n", top_corr.round(4))