# TMU MRP 2025
# XGB Standalone Model
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
    )

X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_val = pd.read_csv("y_val.csv").squeeze()

# Calculate scale pos weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Train Enhanced XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=1200,
    max_depth=9,
    learning_rate=0.015,
    subsample=0.92,
    colsample_bytree=0.92,
    gamma=3,
    min_child_weight=4,
    reg_alpha=4,
    reg_lambda=8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='auc',
    max_bin=256,
    random_state=42,
    n_jobs=-1
    )
xgb_model.fit(X_train, y_train)
y_pred_prob = xgb_model.predict_proba(X_val)[:, 1]

# Checking to find best Threshold
best_f1 = 0
best_threshold = 0.5
for t in np.arange(0.20, 0.51, 0.01):
    preds = (y_pred_prob > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Evaluation Metrics
y_pred = (y_pred_prob > best_threshold).astype(int)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_prob)
pr_auc = average_precision_score(y_val, y_pred_prob)
conf_mat = confusion_matrix(y_val, y_pred)
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")
print(f"PR-AUC:      {pr_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_mat)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='navy')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'PR Curve (AUC = {pr_auc:.4f})', color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBoost Model')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()