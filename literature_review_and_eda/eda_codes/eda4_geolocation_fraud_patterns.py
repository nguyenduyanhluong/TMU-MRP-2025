# TMU MRP 2025
# EDA 4 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt

transaction_df = pd.read_csv('cleaned_train_transaction.csv')

min_threshold = 500

# Figure 7: Fraud Rate by Region (addr1)
addr1_stats = transaction_df.groupby("addr1")["isFraud"].agg(["count", "mean"]).reset_index()
addr1_stats.columns = ["addr1", "count", "fraud_rate"]
addr1_stats = addr1_stats[addr1_stats["count"] > min_threshold].sort_values("fraud_rate", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(addr1_stats["addr1"].astype(str), addr1_stats["fraud_rate"])
plt.xticks(rotation=90)
plt.xlabel("addr1 (Region)", fontsize=20)
plt.ylabel("Fraud Rate", fontsize=20)
plt.title("Fraud Rate by Region (addr1)", fontsize=22)
plt.tight_layout()
plt.show()

# Figure 8: Fraud Rate by Country (addr2)
addr2_stats = transaction_df.groupby("addr2")["isFraud"].agg(["count", "mean"]).reset_index()
addr2_stats.columns = ["addr2", "count", "fraud_rate"]
addr2_stats = addr2_stats[addr2_stats["count"] > min_threshold].sort_values("fraud_rate", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(addr2_stats["addr2"].astype(str), addr2_stats["fraud_rate"])
plt.xlabel("addr2 (Country)", fontsize=14)
plt.ylabel("Fraud Rate", fontsize=14)
plt.title("Fraud Rate by Country (addr2)", fontsize=16)
plt.tight_layout()
plt.show()

# Create a Foreign Transaction Flag
transaction_df["foreign_transaction_flag"] = (transaction_df["addr2"] != 87).astype(int)
foreign_stats = transaction_df.groupby("foreign_transaction_flag")["isFraud"].agg(["count", "mean"]).reset_index()
foreign_stats.columns = ["foreign_flag", "count", "fraud_rate"]
print("\nFraud Rate by Foreign Transaction Flag\n", foreign_stats)