#!/usr/bin/env python3
"""
Credit Card Fraud Detection Pipeline
------------------------------------
Author: [Your Name]
Description: 
    Detects fraudulent transactions in highly imbalanced datasets.
    Implements Logistic Regression, Random Forest, and XGBoost.
    Uses SMOTE for data balancing and Stratified K-Fold for validation.

Usage:
    python fraud_detection.py --data creditcard.csv --model xgb --smote
"""

import argparse
import sys
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.pipeline import Pipeline as SkPipeline

# Try importing XGBoost, fall back if not available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Imbalanced-learn is required for SMOTE Pipeline integration
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: 'imbalanced-learn' is not installed. Run: pip install imbalanced-learn")
    sys.exit(1)

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
IMG_DIR = 'model_outputs'

def setup_dirs():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

def load_data(path):
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        sys.exit(1)
    
    print(f"[Data] Loading {path}...")
    df = pd.read_csv(path)
    
    # Basic data check
    if 'Class' not in df.columns:
        print("[Error] Dataset does not contain 'Class' column.")
        sys.exit(1)

    print(f"[Data] Loaded {df.shape[0]} transactions.")
    print(f"[Data] Class Distribution:\n{df['Class'].value_counts(normalize=True)}")
    return df

def preprocess_data(df):
    """
    Scales 'Amount' and 'Time'. V1-V28 are already PCA transformed.
    Using RobustScaler is better for 'Amount' due to outliers.
    """
    print("[Preprocessing] Scaling Time and Amount features...")
    
    # Scale Amount and Time
    rob_scaler = RobustScaler()
    
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # Move Class to end for cleanliness
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    return df

def get_model(model_name, random_state=42):
    if model_name == 'lr':
        return LogisticRegression(solver='liblinear', random_state=random_state)
    elif model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    elif model_name == 'xgb':
        if HAS_XGB:
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        else:
            print("[Warning] XGBoost not found, falling back to GradientBoosting.")
            return GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError("Unknown model. Choose: lr, rf, xgb")

def build_pipeline(model, use_smote=True):
    """
    Constructs a pipeline.
    CRITICAL: SMOTE must be inside the pipeline so it only oversamples 
    training folds during cross-validation, preventing data leakage.
    """
    steps = []
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    
    steps.append(('model', model))
    
    # Use ImbPipeline if SMOTE is involved, otherwise standard Sklearn Pipeline
    if use_smote:
        return ImbPipeline(steps)
    return SkPipeline(steps)

def evaluate_model(y_test, y_pred, y_prob, model_name):
    print("\n" + "="*40)
    print(f"RESULTS FOR: {model_name.upper()}")
    print("="*40)
    
    # 1. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"{IMG_DIR}/cm_{model_name}.png")
    plt.close()
    
    # 3. ROC-AUC
    roc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_score:.4f}")
    
    # 4. AUPRC (Average Precision) - Vital for Imbalanced Data
    pr_score = average_precision_score(y_test, y_prob)
    print(f"Average Precision-Recall Score: {pr_score:.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_score:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f"{IMG_DIR}/roc_{model_name}.png")
    plt.close()
    
    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f"{model_name} (AP={pr_score:.2f})")
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f"{IMG_DIR}/pr_{model_name}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to creditcard.csv')
    parser.add_argument('--model', type=str, default='lr', choices=['lr', 'rf', 'xgb'], help='Model type')
    parser.add_argument('--smote', action='store_true', help='Apply SMOTE oversampling')
    args = parser.parse_args()
    
    setup_dirs()
    
    # 1. Load
    df = load_data(args.data)
    
    # 2. Preprocess
    df = preprocess_data(df)
    
    # 3. Split Features/Target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 4. Stratified Split (Crucial for imbalance)
    print("[Split] Performing Stratified Train-Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 5. Build Pipeline
    print(f"[Model] Initializing {args.model.upper()} with SMOTE={args.smote}...")
    model = get_model(args.model)
    pipeline = build_pipeline(model, use_smote=args.smote)
    
    # 6. Train
    print("[Training] Fitting model (this may take a while for RF/XGB)...")
    pipeline.fit(X_train, y_train)
    
    # 7. Predictions
    y_pred = pipeline.predict(X_test)
    
    # Get probabilities for AUC/Curves
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = pipeline.decision_function(X_test)
        
    # 8. Evaluation
    evaluate_model(y_test, y_pred, y_prob, args.model)
    
    print(f"\n[Done] Check the '{IMG_DIR}' folder for plots.")

if __name__ == "__main__":
    main()