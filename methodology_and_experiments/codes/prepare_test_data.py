# TMU MRP 2025
# Prepare Test Data Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

test_transaction = pd.read_csv("test_transaction.csv")
test_identity = pd.read_csv("test_identity.csv")

# Merge Data Files
test_full = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")
test_ids = test_full["TransactionID"]

X_test = test_full.drop(columns=["TransactionID"])

categorical_cols = X_test.select_dtypes(include="object").columns
for col in categorical_cols:
    X_test[col] = X_test[col].fillna("missing")
    label_encoder = LabelEncoder()
    try:
        X_test[col] = label_encoder.fit_transform(X_test[col])
    except:
        X_test[col] = 0 

numeric_cols = X_test.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy="median")
X_test[numeric_cols] = imputer.fit_transform(X_test[numeric_cols])

if "TransactionDT" in X_test.columns:
    X_test["hour_of_day"] = (test_full["TransactionDT"] // 3600) % 24

scaler = StandardScaler()
X_test[numeric_cols] = scaler.fit_transform(X_test[numeric_cols])

test_ids.to_csv("test_ids.csv", index=False)
X_test.to_csv("X_test.csv", index=False)