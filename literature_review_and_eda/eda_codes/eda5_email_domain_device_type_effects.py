# TMU MRP 2025
# EDA 5 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt

transaction_df = pd.read_csv("cleaned_train_transaction.csv")
identity_df = pd.read_csv("cleaned_train_identity.csv")

merged_df = pd.merge(
    transaction_df,
    identity_df,
    how='left',
    on='TransactionID'
)

# Figure 9: Fraud Rate by Email Domain (P emaildomain)
top_domains = merged_df['P_emaildomain'].value_counts().head(10).index
merged_df['P_emaildomain_grouped'] = merged_df['P_emaildomain'].apply(lambda x: x if x in top_domains else 'other')

email_stats = merged_df.groupby('P_emaildomain_grouped')['isFraud'].agg(['count', 'mean']).reset_index()
email_stats.columns = ['email_domain', 'count', 'fraud_rate']
email_stats = email_stats.sort_values('fraud_rate', ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(email_stats['email_domain'], email_stats['fraud_rate'])
plt.xticks(rotation=45)
plt.xlabel('Purchaser Email Domain', fontsize=16)
plt.ylabel('Fraud Rate', fontsize=16)
plt.title('Fraud Rate by Email Domain (P_emaildomain)', fontsize=18)
plt.tight_layout()
plt.show()

# Figure 10: Fraud Rate by Device Type
device_type_stats = merged_df.groupby('DeviceType')['isFraud'].agg(['count', 'mean']).reset_index()
device_type_stats.columns = ['DeviceType', 'count', 'fraud_rate']

plt.figure(figsize=(6, 4))
plt.bar(device_type_stats['DeviceType'], device_type_stats['fraud_rate'])
plt.xlabel('Device Type')
plt.ylabel('Fraud Rate')
plt.title('Fraud Rate by Device Type')
plt.tight_layout()
plt.show()

# Figure 11: Fraud Rate by Top 10 DeviceInfo Values
top_devices = merged_df['DeviceInfo'].value_counts().head(10).index
device_info_stats = merged_df[merged_df['DeviceInfo'].isin(top_devices)]
device_info_stats = device_info_stats.groupby('DeviceInfo')['isFraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
device_info_stats.plot(kind='bar')
plt.xlabel('DeviceInfo', fontsize=17)
plt.ylabel('Fraud Rate', fontsize=17)
plt.title('Fraud Rate by Top 10 DeviceInfo', fontsize=20)
plt.tight_layout()
plt.show()