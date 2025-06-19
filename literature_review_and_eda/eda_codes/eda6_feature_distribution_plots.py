# TMU MRP 2025
# EDA 6 Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_train_transaction.csv")

# Figure 13 and 14: Distribution of V45 and V44 by Fraud Class
top_features = ["V45", "V44"]

df_sampled = df.sample(frac=0.1, random_state=1)

for feature in top_features:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df_sampled[df_sampled["isFraud"] == 0], x=feature, label="Not Fraud", fill=True, common_norm=False)
    sns.kdeplot(data=df_sampled[df_sampled["isFraud"] == 1], x=feature, label="Fraud", fill=True, common_norm=False, color="red")
    plt.title(f"Distribution of {feature} by Fraud Class", fontsize=16)
    plt.xlim(0, 6) 
    plt.xlabel(feature, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()