# TMU MRP 2025
# Full Training Data Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_transaction = pd.read_csv("train_transaction.csv")
train_identity = pd.read_csv("train_identity.csv")

# Merge Data Files
train_full = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")

transaction_ids = train_full["TransactionID"]

y_full = train_full["isFraud"]
X_full = train_full.drop(columns=["TransactionID", "isFraud"])

categorical_cols = X_full.select_dtypes(include="object").columns
for col in categorical_cols:
    X_full[col] = X_full[col].fillna("missing")
    le = LabelEncoder()
    X_full[col] = le.fit_transform(X_full[col])

numeric_cols = X_full.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy="median")
X_full[numeric_cols] = imputer.fit_transform(X_full[numeric_cols])

if "TransactionDT" in X_full.columns:
    X_full["hour_of_day"] = (train_full["TransactionDT"] // 3600) % 24

scaler = StandardScaler()
X_full[numeric_cols] = scaler.fit_transform(X_full[numeric_cols])

X_full.to_csv("X_full.csv", index=False)
y_full.to_csv("y_full.csv", index=False)