# TMU MRP 2025
# Random Forest Feature Importance
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
    )
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_names = X_train.columns

# Get top 20 features
top_indices = np.argsort(importances)[::-1][:20]
top_features = feature_names[top_indices]
top_importances = importances[top_indices]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.xlabel("Feature Importance", fontsize=14)
plt.title("Top 20 Feature Importances (Random Forest)", fontsize=16)
plt.tight_layout()
plt.show()