# TMU MRP 2025
# Preprocessing Code File
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import os

def preprocessing(file_path):
    data_file = pd.read_csv(file_path)
    data_file.columns = [col.strip().replace("-", "_") for col in data_file.columns]

    for col in data_file.columns:
        if data_file[col].dtype == 'object':
            data_file[col] = data_file[col].fillna('missing')
        else:
            data_file[col] = data_file[col].where(pd.notnull(data_file[col]), None)

    return data_file

base_path = r"C:\Users\Andy's PC\OneDrive\Desktop\MRP 2025\Code Files - Literature Review and EDA"

train_transaction_cleaned = preprocessing(os.path.join(base_path, "train_transaction.csv"))
train_transaction_cleaned.to_csv(os.path.join(base_path, "cleaned_train_transaction.csv"), index=False)

train_identity_cleaned = preprocessing(os.path.join(base_path, "train_identity.csv"))
train_identity_cleaned.to_csv(os.path.join(base_path, "cleaned_train_identity.csv"), index=False)