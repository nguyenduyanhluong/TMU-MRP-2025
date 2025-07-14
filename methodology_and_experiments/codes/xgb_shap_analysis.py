# TMU MRP 2025
# XGB SHAP Analysis
# Student Name: Nguyen Duy Anh Luong
# Student ID: 500968520

import pandas as pd
import xgboost as xgb
import shap

X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_val = pd.read_csv("y_val.csv").squeeze()

# Train XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=27.58,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
    )
model.fit(X_train, y_train)

explainer = shap.Explainer(model)
sample_X = X_val.sample(n=1000, random_state=42)
shap_values = explainer(sample_X)

# Plot SHAP
shap.summary_plot(shap_values, sample_X, show=True)
shap.summary_plot(shap_values, sample_X, plot_type="bar", show=True)