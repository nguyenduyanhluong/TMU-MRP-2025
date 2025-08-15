# TMU MRP 2025
# Random Forest, DNN, Enhanced XGBoost, XGB + DNN Ensemble ROC and PR-AUC Curve Comparison
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
    )
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

X_train = pd.read_csv("X_train.csv")
X_val   = pd.read_csv("X_val.csv")
y_train = pd.read_csv("y_train.csv").squeeze().astype(int)
y_val   = pd.read_csv("y_val.csv").squeeze().astype(int)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
    )
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_val)[:, 1]

# DNN
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled   = scaler.transform(X_val_imp)

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow((1 - p_t), gamma)
        return -K.mean(alpha_factor * modulating_factor * K.log(p_t))
    return loss

dnn_model = Sequential([
    Dense(512, input_shape=(X_train_scaled.shape[1],)),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128),
    LeakyReLU(alpha=0.01),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
    ])
dnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=["accuracy"]
    )
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
dnn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=20,
    batch_size=1024,
    callbacks=[early_stop, reduce_lr],
    verbose=2
    )
dnn_probs = dnn_model.predict(X_val_scaled).ravel()

# Enhanced XGB
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_adv_model = XGBClassifier(
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
xgb_adv_model.fit(X_train, y_train)
xgb_adv_probs = xgb_adv_model.predict_proba(X_val)[:, 1]

# XGB + DNN Ensemble
xgb_probs = xgb_adv_model.predict_proba(X_val)[:, 1]

X_train_fill = X_train.fillna(X_train.median(numeric_only=True))
X_val_fill   = X_val.fillna(X_train.median(numeric_only=True))

scaler_ens = StandardScaler()
X_train_scaled_ens = scaler_ens.fit_transform(X_train_fill)
X_val_scaled_ens   = scaler_ens.transform(X_val_fill)

c_w = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
class_weights = dict(enumerate(c_w))

dnn_ens_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled_ens.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
    ])
dnn_ens_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=focal_loss(gamma=2),
    metrics=['AUC', 'Precision', 'Recall']
    )
early_stop_ens = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
dnn_ens_model.fit(
    X_train_scaled_ens, y_train,
    validation_data=(X_val_scaled_ens, y_val),
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop_ens],
    verbose=1
    )
dnn_ens_probs = dnn_ens_model.predict(X_val_scaled_ens).flatten()

ensemble_probs = 0.6 * xgb_probs + 0.4 * dnn_ens_probs

# Plot ROC curves
plt.figure(figsize=(8, 6))
for name, probs in [
    ("Random Forest", rf_probs),
    ("DNN", dnn_probs),
    ("XGB-Adv", xgb_adv_probs),
    ("XGB + DNN Ensemble", ensemble_probs),]:
    fpr, tpr, _ = roc_curve(y_val, probs)
    auc_val = roc_auc_score(y_val, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — RF & DNN Baselines vs Proposed Models")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()

# Plot Precision–Recall curves
plt.figure(figsize=(8, 6))
for name, probs in [
    ("Random Forest", rf_probs),
    ("DNN", dnn_probs),
    ("XGB-Adv", xgb_adv_probs),
    ("XGB + DNN Ensemble", ensemble_probs),]:
    precision, recall, _ = precision_recall_curve(y_val, probs)
    ap_val = average_precision_score(y_val, probs)
    plt.plot(recall, precision, lw=2, label=f"{name} (AP = {ap_val:.4f})")

plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curves — RF & DNN Baselines vs Proposed Models")
plt.legend(loc="lower left", fontsize=8)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()