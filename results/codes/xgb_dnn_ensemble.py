# TMU MRP 2025
# XGB + DNN Ensemble Model
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
    )
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

# Focal Loss
def focal_loss(gamma = 2.0, alpha = 0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * K.pow(1. - p_t, gamma) * K.log(p_t)
        return K.mean(loss)
    return loss

X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train = pd.read_csv("y_train.csv").squeeze().astype(int)
y_val = pd.read_csv("y_val.csv").squeeze().astype(int)

X_train = X_train.fillna(X_train.median(numeric_only=True))
X_val = X_val.fillna(X_train.median(numeric_only=True))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Enhanced XGB Model
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
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
xgb_probs = xgb_model.predict_proba(X_val)[:, 1]

# DNN Model
c_w = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(c_w))

input_dim = X_train_scaled.shape[1]
dnn_model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
    ])

dnn_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=focal_loss(gamma=2),
    metrics=['AUC', 'Precision', 'Recall']
    )

early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
dnn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
    )
dnn_probs = dnn_model.predict(X_val_scaled).flatten()

# Ensemble Strategy
ensemble_probs = 0.6 * xgb_probs + 0.4 * dnn_probs

# Checking to find best Threshold
best_f1 = 0
best_threshold = 0.5
for t in np.arange(0.1, 0.6, 0.01):
    preds = (ensemble_probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Evaluation Metrics
final_preds = (ensemble_probs >= best_threshold).astype(int)
conf_mat = confusion_matrix(y_val, final_preds)
print(f"ROC-AUC:     {roc_auc_score(y_val, ensemble_probs):.4f}")
print(f"PR-AUC:      {average_precision_score(y_val, ensemble_probs):.4f}")
print(f"F1 Score:    {f1_score(y_val, final_preds):.4f}")
print(f"Precision:   {precision_score(y_val, final_preds):.4f}")
print(f"Recall:      {recall_score(y_val, final_preds):.4f}")
print("\nConfusion Matrix:")
print(conf_mat)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_val, ensemble_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_val, ensemble_probs):.4f})', color='navy')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGB + DNN Ensemble')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_val, ensemble_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'PR Curve (AUC = {average_precision_score(y_val, ensemble_probs):.4f})', color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGB + DNN Ensemble')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()