# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)

## 📌 Project Overview
This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions. The dataset is highly imbalanced, with positive class (frauds) accounting for only **0.172%** of all transactions.

Key techniques used:
- **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance.
- **Stratified K-Fold Validation** to ensure reliable performance metrics.
- **Robust Scaling** for handling outliers in transaction amounts.

## 📊 Key Results
The model was optimized to maximize **Precision-Recall (AUPRC)** rather than simple accuracy.

| Model | ROC-AUC | PR-AUC (Average Precision) |
| :--- | :--- | :--- |
| **XGBoost** | **0.98** | **0.87** |
| Random Forest | 0.96 | 0.85 |
| Logistic Regression | 0.94 | 0.78 |

*(Note: Replace these numbers with your actual results)*

## 📈 Visualizations
### Confusion Matrix
![Confusion Matrix](figures/cm_xgb.png)

### Precision-Recall Curve
![PR Curve](figures/pr_xgb.png)

## 🛠️ Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection