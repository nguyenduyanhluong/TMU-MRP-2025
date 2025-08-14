# TMU-MRP-2025
# DEVELOPMENT OF MACHINE LEARNING-BASED ALGORITHM FOR ENHANCING TRANSACTION FRAUD DETECTION
## Overview

This repository contains the implementation and experiments from my **Major Research Project (MRP)** for the MSc in Data Science and Analytics program at Toronto Metropolitan University. The project focuses on improving credit card fraud detection using supervised machine learning, comparing both single-model and hybrid approaches, and evaluating their performance on a real-world dataset.

The study explored multiple algorithms, identified the most effective ones, and examined their decision-making process using model interpretability techniques. Recommendations are provided for deploying these models in practical fraud detection systems.

## Objectives

- Identify the supervised ML model that offers the best balance between detection accuracy and false positives.
- Test the benefits of combining models in an ensemble compared to using individual models.
- Explore the role of feature engineering in improving model accuracy and robustness.
- Handle class imbalance effectively using **class weighting.**
- Evaluate models using appropriate metrics for imbalanced data such as ROC-AUC, PR-AUC, F1-score, recall, and precision.

## Methodology
- **Model Selection:** Evaluated Logistic Regression, Random Forest, XGBoost, Artificial Neural Networks, Deep Neural Networks, and an **XGBoost + DNN ensemble.**
- **Class Imbalance Handling:** Applied **class weighting** to improve detection of rare fraud cases.
- **Ensemble Approach:** Combined XGBoost with DNN through weighted probability averaging to leverage the strengths of both.
- **Evaluation Metrics:** Measured ROC-AUC, PR-AUC, precision, recall, F1-score, and confusion matrices.
- **Model Interpretation:** Used SHAP for XGBoost and feature importance for Random Forest to understand which features drive fraud detection.

## Key Results
- **Enhanced XGBoost and XGBoost + DNN Ensemble** delivered the best overall performance.
- The **ensemble model** achieved strong precision–recall balance, making it suitable for minimizing both missed fraud and false positives.
- Feature analysis revealed that transaction amount, card details, engineered behavioral features, and time-based patterns are key predictors.
- No identity-based features (e.g., device type, email domain) were added, as they were unnecessary for this dataset.

## Dataset

**Source:** [IEEE-CIS Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The dataset contains anonymized transaction information with severe class imbalance, making it ideal for testing fraud detection methods.

## Contact

**Author:** Nguyen Duy Anh Luong  
**Supervisor:** Dr. Shengkun Xie  
**Email:** [nguyenduyanh.luong@torontomu.ca]
