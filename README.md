# TMU-MRP-2025
# Optimizing Supervised Machine Learning for Enhanced Transaction Fraud Detection Accuracy
## Results
## Overview

This repository contains the final phase of my Major Research Project (MRP) for the MSc in Data Science and Analytics program at Toronto Metropolitan University. This stage focuses on optimizing the best-performing models from the Methodology and Experiments phase, comparing their performance against real-world benchmarks, and providing recommendations for practical deployment in fraud detection systems.

## Objectives

- Refine and optimize top-performing fraud detection models
- Compare standalone XGBoost and a hybrid XGBoost + Deep Neural Network (DNN) ensemble
- Evaluate models using metrics suited for imbalanced data (ROC-AUC, PR-AUC, F1-score, recall, precision)
- Benchmark performance against the winning Kaggle IEEE-CIS Fraud Detection solution
- Provide actionable insights for deploying fraud detection models in production environments

## Methodology
- **Model Optimization:** Tuned XGBoost with advanced hyperparameters (regularization, learning rate, feature sampling) for maximum recall and balanced performance.
- **Ensemble Approach:** Combined XGBoost with a DNN using a weighted probability average (0.6 for XGBoost, 0.4 for DNN) to leverage the strengths of both models.
- **Loss Function:** Implemented Focal Loss in the DNN to handle extreme class imbalance and reduce the impact of easily classified examples.
- **Evaluation Strategy:** Compared models using validation data and reported metrics like ROC-AUC, PR-AUC, precision, recall, F1-score, and confusion matrices.
- **Benchmarking:** Measured performance against the 1st place Kaggle model to assess competitiveness.

## Key Results
- The Enhanced XGBoost model achieved a ROC-AUC of 0.9628 and a PR-AUC of 0.7676, identifying over 82% of fraud cases with a reasonable trade-off on precision.
- The XGBoost + DNN ensemble reached a ROC-AUC of 0.9629 and a PR-AUC of 0.7898, with a precision of 0.7818 and recall of 0.6927, striking a strong balance between catching fraud and limiting false positives.
- Both models outperformed the standalone 1st place XGBoost from the Kaggle competition (ROC-AUC 0.9602 public, 0.9324 private).
- The ensemble’s performance makes it especially promising for deployment, where minimizing missed fraud and excessive false alarms is critical.

## Dataset

**Source:** [IEEE-CIS Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

This dataset contains anonymized transaction and identity information from real-world e-commerce activity, reflecting challenges such as class imbalance, anonymized features, and time-based behavior.

## Contact

**Author:** Nguyen Duy Anh Luong  
**Supervisor:** Dr. Shengkun Xie  
**Email:** [nguyenduyanh.luong@torontomu.ca]

## Next Steps

The next and final stage of this project will be the **Final Report**, which will:
- Summarize the findings across all project phases
- Compare all models and results in one place
- Provide final recommendations for deploying the fraud detection system in a real-world setting
- Document lessons learned and opportunities for further research
