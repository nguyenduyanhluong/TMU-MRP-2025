# TMU MRP 2025
# EDA 7 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_train_transaction.csv")

# Figure 15: Fraud Rate by card4
card4_stats = df.groupby("card4")["isFraud"].mean().sort_values(ascending=False)
card4_stats.plot(kind="bar", title="Fraud Rate by card4", ylabel="Fraud Rate", xlabel="card4", figsize=(6, 4))
plt.tight_layout()
plt.show()

# Figure 16: Fraud Rate by Top 10 card5 Values
top_card5 = df["card5"].value_counts().head(10).index
card5_df = df[df["card5"].isin(top_card5)]
card5_stats = card5_df.groupby("card5")["isFraud"].mean().sort_values(ascending=False)
card5_stats.plot(kind="bar", title="Fraud Rate by Top 10 card5 Values", ylabel="Fraud Rate", xlabel="card5", figsize=(6, 4))
plt.tight_layout()
plt.show()

# Figure 17: Fraud Rate by M6
df["M6"] = df["M6"].fillna("missing") 
m6_stats = df.groupby("M6")["isFraud"].mean().sort_values(ascending=False)
m6_stats.plot(kind="bar", title="Fraud Rate by M6", ylabel="Fraud Rate", xlabel="M6", figsize=(6, 4))
plt.tight_layout()
plt.show()