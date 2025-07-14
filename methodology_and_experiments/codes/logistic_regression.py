# TMU MRP 2025
# Logistic Regression Model
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc
    )

X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_val = pd.read_csv("y_val.csv").squeeze()

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Build LR Model
model = LogisticRegression(
    solver="liblinear",
    class_weight="balanced",
    max_iter=200,
    random_state=42
    )
model.fit(X_train_imputed, y_train)

y_pred = model.predict(X_val_imputed)
y_proba = model.predict_proba(X_val_imputed)[:, 1]

# Evaluation Metrics
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
roc_auc = roc_auc_score(y_val, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

precision, recall, _ = precision_recall_curve(y_val, y_proba)
pr_auc = auc(recall, precision)
print(f"PR-AUC Score: {pr_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.title("Precision-Recall Curve - Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()