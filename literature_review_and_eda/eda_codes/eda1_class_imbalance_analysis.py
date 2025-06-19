# TMU MRP 2025
# EDA 1 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib.patches import Patch

transaction_df = pd.read_csv('cleaned_train_transaction.csv')

# Figure 1: Fraud Distribution by Transaction Type
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=transaction_df, x='isFraud', palette='pastel')

plt.title('Class Imbalance in Fraud Detection Dataset', fontsize=18)
plt.xlabel('Transaction Type', fontsize=14)
plt.ylabel('Number of Transactions', fontsize=14)
ax.set_xticklabels(['Legitimate', 'Fraudulent'])

legend_labels = [Patch(color=ax.patches[0].get_facecolor(), label='Legitimate'), Patch(color=ax.patches[1].get_facecolor(), label='Fraudulent')]
plt.legend(handles=legend_labels, title='isFraud', loc='upper right')

total = len(transaction_df)
for p in ax.patches:
    height = p.get_height()
    percentage = f'{100 * height / total:.2f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=11, color='black')

plt.tight_layout()
plt.show()

# Class Ratio Calculation
fraud_count = transaction_df['isFraud'].value_counts()
fraud_ratio = fraud_count[1] / fraud_count.sum()

print("Class Distribution Summary")
print(f"Legitimate transactions: {fraud_count[0]}")
print(f"Fraudulent transactions: {fraud_count[1]}")
print(f"Fraud ratio: {fraud_ratio:.4f} ({fraud_ratio * 100:.2f}%)\n")

# Dummy Classifier Benchmark
X_dummy = transaction_df.drop(columns=['isFraud'])
y = transaction_df['isFraud']
X_dummy_filled = X_dummy.fillna(0)

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_dummy_filled, y)
y_pred = dummy.predict(X_dummy_filled)

print("Dummy Classifier Performance")
print(f"Accuracy:  {dummy.score(X_dummy_filled, y):.4f}")
print(f"Precision: {precision_score(y, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y, y_pred):.4f}")
print(f"F1-score:  {f1_score(y, y_pred):.4f}\n")

# Business Cost Estimation
cost_per_fraud = 500
estimated_loss = fraud_count[1] * cost_per_fraud
print("Estimated Financial Impact if Fraud is Missed")
print(f"Potential loss if model misses all fraud: ${estimated_loss:,}")