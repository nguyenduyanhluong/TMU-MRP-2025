# TMU MRP 2025
# EDA 9 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_train_transaction.csv")

def plot_binned_fraud_rate(df, column, bins, title, xlabel):
    bin_col = f"{column}_bin"
    df[bin_col] = pd.cut(df[column], bins=bins, include_lowest=True)
    stats = df.groupby(bin_col)["isFraud"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=bin_col, y="isFraud", data=stats)
    plt.title(title, fontsize=16)
    plt.ylabel("Fraud Rate", fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Figure 21: Fraud Rate by Distance Range (dist1)
plot_binned_fraud_rate(
    df,
    column="dist1",
    bins=[0, 50, 100, 200, 500, 1000, 5000, df["dist1"].max()],
    title="Fraud Rate by Distance (dist1)",
    xlabel="Distance Range (dist1)"
)

# Figure 22: Fraud Rate by Distance Range (dist2)
plot_binned_fraud_rate(
    df,
    column="dist2",
    bins=[0, 50, 100, 200, 500, 1000, 5000, df["dist2"].max()],
    title="Fraud Rate by Distance (dist2)",
    xlabel="Distance Range (dist2)"
)

# Figure 23: Fraud Rate by Count Range (C13)
plot_binned_fraud_rate(
    df,
    column="C13",
    bins=[0, 1, 2, 3, 5, 10, 20, 50, df["C13"].max()],
    title="Fraud Rate by Count (C13)",
    xlabel="C13 Count Range"
)