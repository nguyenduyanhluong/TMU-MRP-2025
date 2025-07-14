# TMU-MRP-2025
# Optimizing Supervised Machine Learning for Enhanced Transaction Fraud Detection Accuracy
## Methodology and Experiments
## Overview

This repository contains the third phase of my Major Research Project (MRP) for the MSc in Data Science and Analytics program at Toronto Metropolitan University. This stage focuses on building, training, and evaluating supervised machine learning models for credit card fraud detection using the IEEE-CIS Fraud Detection dataset.

## Objectives

- Develop supervised machine learning models to classify transactions as fraudulent or non-fraudulent
- Handle extreme class imbalance using model-based techniques (e.g., class weighting, focal loss)
- Evaluate models using metrics suited for imbalanced data, including ROC-AUC, PR-AUC, F1-score, recall, and precision
- Interpret model decisions using SHAP values and feature importance plots to support explainability

## Methodology
- **Data Preprocessing:** Combined transaction and identity datasets, handled missing values, label-encoded categorical variables, extracted time-based features, and split the data with stratification
- **Class Imbalance Handling:** Applied scale-sensitive class weights and focal loss to give more importance to fraud cases without oversampling
- **Model Development:** Trained and evaluated Logistic Regression, Random Forest, XGBoost, Artificial Neural Network (ANN), and Deep Neural Network (DNN)
- **Evaluation Strategy:** Assessed models on accuracy, recall, precision, F1-score, ROC-AUC, PR-AUC, and confusion matrices
- **Model Interpretation:** Used SHAP (for XGBoost) and built-in feature importance (for Random Forest) to analyze which features most influence predictions

## Dataset

**Source:** [IEEE-CIS Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

This dataset contains anonymized transaction and identity information from real-world e-commerce activity, reflecting challenges such as class imbalance, anonymized features, and time-based behavior.

## Contact

**Author:** Nguyen Duy Anh Luong  
**Supervisor:** Dr. Shengkun Xie  
**Email:** [nguyenduyanh.luong@torontomu.ca]

## Next Steps

The next stage of this project will focus on **Results**:
- Model optimization and ensemble techniques (e.g., XGBoost Stand Alone, XGBoost + DNN)
- Final evaluation and comparison
- Insights and recommendations for real-world deployment of fraud detection models
