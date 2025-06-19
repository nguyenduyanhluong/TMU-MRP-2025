# TMU MRP 2025
# EDA 3 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt

transaction_df = pd.read_csv('cleaned_train_transaction.csv')

# Create time-based features
transaction_df['Transaction_day'] = transaction_df['TransactionDT'] // (60 * 60 * 24)
transaction_df['Transaction_hour'] = (transaction_df['TransactionDT'] // (60 * 60)) % 24
transaction_df['is_weekend'] = (transaction_df['Transaction_day'] % 7 >= 5).astype(int)

# Figure 5: Fraud Rate by Hour of Day
fraud_by_hour = transaction_df.groupby('Transaction_hour')['isFraud'].mean()

plt.figure(figsize=(10, 5))
plt.plot(fraud_by_hour.index, fraud_by_hour.values, marker='o', label='Fraud Rate')
plt.title('Fraud Rate by Hour of Day', fontsize=18)
plt.xlabel('Hour of Day (0–23)', fontsize=16)
plt.ylabel('Fraud Rate', fontsize=16)
plt.grid(True)
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.show()

# Compare fraud rate between weekday and weekend
fraud_by_weekend = transaction_df.groupby('is_weekend')['isFraud'].mean()
count_by_weekend = transaction_df['is_weekend'].value_counts()

print("Fraud Rate by Weekend Indicator")
print(f"Weekday (0): {count_by_weekend[0]} transactions, Fraud Rate = {fraud_by_weekend[0]:.4f}")
print(f"Weekend (1): {count_by_weekend[1]} transactions, Fraud Rate = {fraud_by_weekend[1]:.4f}")

# Figure 6: Fraud Rate by Transaction Day
fraud_by_day = transaction_df.groupby('Transaction_day')['isFraud'].mean()

plt.figure(figsize=(12, 4))
plt.plot(fraud_by_day.index, fraud_by_day.values, color='purple', label='Fraud Rate')
plt.title('Fraud Rate by Transaction Day', fontsize=20)
plt.xlabel('Transaction Day', fontsize=18)
plt.ylabel('Fraud Rate', fontsize=18)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()