# TMU MRP 2025
# Preprocessing Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_transaction = pd.read_csv("train_transaction.csv")
train_identity = pd.read_csv("train_identity.csv")

# Replace - with _
train_transaction.columns = train_transaction.columns.str.replace('-', '_')
train_identity.columns = train_identity.columns.str.replace('-', '_')

# Fill Missing Values
for col in train_transaction.select_dtypes(include='object').columns:
    train_transaction[col] = train_transaction[col].fillna('missing')

for col in train_identity.select_dtypes(include='object').columns:
    train_identity[col] = train_identity[col].fillna('missing')

train_transaction.to_csv("cleaned_train_transaction.csv", index=False)
train_identity.to_csv("cleaned_train_identity.csv", index=False)

# Merge Data Files
df = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")

categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

if "TransactionDT" in df.columns:
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24

X = df.drop(columns=["isFraud", "TransactionID"])
y = df["isFraud"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)