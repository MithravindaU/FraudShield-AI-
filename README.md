# FraudShield AI 💳🛡️

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blueviolet?style=for-the-badge)
![Random Forest](https://img.shields.io/badge/Random-Forest-success?style=for-the-badge)
![Fraud Detection](https://img.shields.io/badge/Fraud-Detection-orange?style=for-the-badge)

## 🚀 Overview

FraudShield AI is a machine learning-based credit card fraud detection system designed to identify potentially fraudulent transactions from highly imbalanced transaction data.

The project uses a **Random Forest Classifier** and explores the effect of **SMOTE (Synthetic Minority Over-sampling Technique)** on fraud detection performance.

The models are evaluated using Precision, Recall, F1-Score, ROC-AUC, confusion matrices, and feature importance analysis.

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SMOTE
- Matplotlib
- Seaborn
- Google Colab

---

## 📂 Dataset

The project uses a credit card transaction dataset containing:

- Transaction time
- Anonymized numerical features (`V1`–`V28`)
- Transaction amount
- Transaction class

### Target Classes

| Class | Meaning |
|------|---------|
| 0 | Normal Transaction |
| 1 | Fraudulent Transaction |

The dataset contains a severe class imbalance:

| Class | Transactions | Percentage |
|------|-------------:|-----------:|
| Normal | 284,315 | 99.83% |
| Fraud | 492 | 0.17% |

This imbalance makes fraud detection particularly challenging.

---

## 🔍 Data Analysis

The project performs exploratory analysis including:

- Class distribution
- Missing value detection
- Duplicate row detection
- Transaction amount distribution
- Transaction time distribution

The dataset contains **1,081 duplicate rows** and no missing values in the analyzed columns.

---

## ⚡ Project Workflow


Credit Card Dataset
        ↓
Data Loading
        ↓
Data Quality Analysis
        ↓
Class Imbalance Analysis
        ↓
Feature Scaling
        ↓
Train-Test Split
        ↓
Random Forest Classifier
        ↓
Model Evaluation
        ↓
SMOTE Oversampling
        ↓
Random Forest with SMOTE
        ↓
Performance Comparison
        ↓
Feature Importance Analysis


---

## 📈 Model Performance
Random Forest vs Random Forest + SMOTE
Metric	Without SMOTE	With SMOTE
Precision	0.941	0.845
Recall	0.816	0.837
F1-Score	0.874	0.841
ROC-AUC	0.963	0.973
Key Observation

SMOTE increased the model's ability to identify fraudulent transactions:

Fraud Recall

Without SMOTE → 81.63%
With SMOTE    → 83.67%

The ROC-AUC also improved:

0.963 → 0.973

However, precision and F1-score decreased slightly after oversampling.

This highlights an important machine learning trade-off between detecting more fraudulent transactions and minimizing false fraud predictions.

---

## 🎯 Confusion Matrix
Random Forest Without SMOTE
                 Predicted
              Normal   Fraud

Actual Normal   56859      5
Actual Fraud       18     80
Random Forest + SMOTE
                 Predicted
              Normal   Fraud

Actual Normal   56849     15
Actual Fraud       16     82

The SMOTE-based model detected 82 of 98 fraudulent transactions in the test set.

---

## 🌟 Feature Importance

The Random Forest model identifies the following as the top contributing features:

Rank	Feature	Importance
1	V14	0.1956
2	V10	0.1105
3	V4	0.1061
4	V12	0.0951
5	V17	0.0850
6	V3	0.0605
7	V11	0.0564
8	V16	0.0438
9	V2	0.0370
10	V9	0.0262

These feature importance values show which anonymized transaction features contributed most strongly to the Random Forest's predictions.

---

## 🌌 Core Concepts Demonstrated
Concept	Purpose
Classification	Identify fraudulent transactions
Random Forest	Build an ensemble fraud classifier
Class Imbalance	Handle rare fraud cases
SMOTE	Oversample the minority class
Feature Scaling	Standardize numerical features
Precision	Measure correctness of fraud predictions
Recall	Measure detected fraudulent transactions
F1-Score	Balance precision and recall
ROC-AUC	Evaluate classification performance
Feature Importance	Interpret model contributions


---


## 📘 Key Learnings
-Machine learning for fraud detection
-Working with highly imbalanced datasets
-Random Forest classification
-SMOTE-based oversampling
-Classification evaluation metrics
-Confusion matrix interpretation
-ROC curve analysis
-Feature importance analysis
-Model comparison

---

## 🔥 Why This Project Matters

Fraud detection is a real-world machine learning problem where fraudulent transactions are usually much rarer than legitimate transactions.

Because of this imbalance, accuracy alone is not enough.

This project demonstrates why metrics such as:

Precision
Recall
F1-Score
ROC-AUC

are important when evaluating fraud detection systems.

The project also demonstrates how changing the class distribution with SMOTE can affect the balance between fraud detection and false positives.

---
## 📊 Visualizations

The project includes:

-Class distribution chart
-Transaction amount distribution
-Transaction time distribution
-Confusion matrix heatmaps
-ROC curve
-Before/after SMOTE comparison
-Model performance comparison
-Top 10 feature importance chart

---
## 🔮 Future Improvements
-Hyperparameter tuning
-Cross-validation
-Threshold optimization
-Precision-Recall curve analysis
-Comparison with Logistic Regression and XGBoost
-Real-time transaction prediction
-Interactive fraud detection dashboard
-Model deployment using Streamlit
-Explainable AI using SHAP


---

## 👩‍💻 Author

Mithravinda U
