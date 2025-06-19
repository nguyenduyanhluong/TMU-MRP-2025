# TMU MRP 2025
# EDA 2 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

transaction_df = pd.read_csv("cleaned_train_transaction.csv")

# Figure 2: Distribution of Transaction Amounts by Fraud Class
plt.figure(figsize=(10, 6))
sns.kdeplot(data=transaction_df[transaction_df['isFraud'] == 0], x='TransactionAmt', label='Legitimate', fill=True)
sns.kdeplot(data=transaction_df[transaction_df['isFraud'] == 1], x='TransactionAmt', label='Fraudulent', fill=True, color='red')
plt.title('Distribution of Transaction Amounts by Fraud Class', fontsize=19)
plt.xlabel('Transaction Amount (USD)', fontsize=17)
plt.ylabel('Density', fontsize=17)
plt.xlim(0, 1300) 
plt.legend()
plt.tight_layout()
plt.show()

# Figure 3: Transaction Amounts by Fraud Class (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=transaction_df, x='isFraud', y='TransactionAmt', palette='pastel')
plt.title('Transaction Amounts by Fraud Class', fontsize=16)
plt.xlabel('Transaction Type', fontsize=14)
plt.ylabel('Transaction Amount (USD)', fontsize=14)
plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
plt.ylim(0, 2000)  
plt.tight_layout()
plt.show()

# Figure 4: Fraud Rate by Product Category (ProductCD)
fraud_rate_by_product = transaction_df.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(7, 5))
sns.barplot(x=fraud_rate_by_product.index, y=fraud_rate_by_product.values, palette='pastel')
plt.title('Fraud Rate by Product Category (ProductCD)', fontsize=16)
plt.xlabel('ProductCD', fontsize=14)
plt.ylabel('Fraud Rate', fontsize=14)
plt.tight_layout()
plt.show()

# Feature Creation: High-Amount Risky Product
transaction_df['amt_over_500_risky_product'] = (
    (transaction_df['TransactionAmt'] > 500) &
    (transaction_df['ProductCD'].isin(['C', 'S', 'H']))
).astype(int)

fraud_rate = transaction_df.groupby('amt_over_500_risky_product')['isFraud'].mean()
count = transaction_df['amt_over_500_risky_product'].value_counts()

print("Fraud Rate for amt_over_500_risky_product (C, S, H)")
print(f"0 = Not high-risk: {count[0]} transactions, Fraud Rate = {fraud_rate[0]:.4f}")
print(f"1 = High-risk:     {count[1]} transactions, Fraud Rate = {fraud_rate[1]:.4f}")