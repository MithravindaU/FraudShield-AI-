# 🛡️ FraudShield AI
### Credit Card Fraud Detection Using Machine Learning

FraudShield AI is a machine learning project that detects potentially fraudulent credit card transactions using a **Random Forest classifier**. It explores how class imbalance affects fraud detection and evaluates whether **Synthetic Minority Over-sampling Technique (SMOTE)** can improve model performance.

The project focuses on model evaluation, fraud detection metrics, and the trade-off between identifying fraudulent transactions and incorrectly flagging legitimate ones.

## 📌 Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a tiny fraction of all transactions. A model can achieve high accuracy simply by predicting most transactions as legitimate, while still missing fraud.

FraudShield AI addresses this challenge by comparing two approaches:

- **Baseline model:** Random Forest trained on the original imbalanced dataset.
- **Resampled model:** Random Forest trained on the training data after applying SMOTE.

Both models are evaluated using the same held-out test set.

## 🎯 Objectives

- Explore and understand credit card transaction data.
- Analyze class imbalance and missing or duplicate records.
- Build a Random Forest classification model.
- Apply SMOTE to address class imbalance in the training data.
- Compare precision, recall, F1-score, and ROC-AUC.
- Visualize model performance and feature importance.
- Understand the trade-offs involved in fraud detection.

## 🗂️ Dataset

**Dataset:** Credit Card Fraud Detection

The project uses a credit card transaction dataset containing anonymized transaction features and a binary target variable.

| Property | Description |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Total input columns | 30 |
| Target column | `Class` |
| Missing values | 0 reported |

The target variable is defined as:

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

Fraudulent transactions account for approximately **0.173%** of the dataset, making this a highly imbalanced classification problem.

The dataset includes `Time`, `Amount`, and anonymized numerical features named `V1` through `V28`.

**Dataset source:** [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset is loaded from Google Drive in the Colab notebook. Download it separately and update the file path to match your environment. The dataset itself is not included in this repository.

## 🧰 Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- Random Forest Classifier
- SMOTE

## ⚙️ Project Workflow

1. **Data loading:** Load the dataset from Google Drive using Pandas.
2. **Exploratory data analysis:** Examine the dataset, class distribution, transaction amounts, and transaction times.
3. **Data quality checks:** Inspect missing values and identify duplicate rows.
4. **Feature preparation:** Separate the input features from the `Class` target variable and scale `Time` and `Amount`.
5. **Train-test split:** Use an 80:20 split with stratification to preserve the class proportions.
6. **Baseline training:** Train a Random Forest classifier on the original training data.
7. **Class imbalance handling:** Apply SMOTE to the training data only.
8. **Resampled model training:** Train a second Random Forest classifier on the resampled training set.
9. **Model evaluation:** Compare classification reports, confusion matrices, ROC-AUC, and other metrics.
10. **Visualization:** Generate ROC curves, model comparison plots, class distribution plots, and feature importance charts.

### Workflow Diagram

```text
Credit Card Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Feature Preparation
        |
        v
Train-Test Split
        |
        +----------------------+
        |                      |
        v                      v
Random Forest            Apply SMOTE
(Baseline)               to Training Data
        |                      |
        v                      v
Baseline Evaluation      Random Forest
                         (SMOTE)
                              |
                              v
                         Model Evaluation
        |                      |
        +----------+-----------+
                   |
                   v
          Performance Comparison
                   |
                   v
        Visualizations and Insights
```

## 🤖 Models Implemented

### 1. Random Forest Classifier

The baseline Random Forest model is trained on the original training dataset without oversampling.

### 2. Random Forest with SMOTE

SMOTE generates synthetic examples of the minority class in the training data to reduce class imbalance. The resulting balanced training dataset is used to train a second Random Forest model.

The test set remains unchanged for both evaluations.

## 📊 Results

The following results were reported by the current notebook before the preprocessing pipeline was corrected for data leakage. They are included to document the existing experiment and **should be regenerated before being treated as final results**.

| Evaluation Metric | Random Forest | Random Forest + SMOTE |
|---|---:|---:|
| Precision | 94.12% | 84.54% |
| Recall | 81.63% | 83.67% |
| F1-score | 87.43% | 84.10% |
| ROC-AUC | 0.9630 | 0.9731 |
| Fraud cases detected | 80 / 98 | 82 / 98 |
| False positives | 5 | 15 |

### Confusion Matrix Results

**Random Forest**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actually legitimate | 56,859 | 5 |
| Actually fraud | 18 | 80 |

**Random Forest + SMOTE**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actually legitimate | 56,849 | 15 |
| Actually fraud | 16 | 82 |

### Key Findings

- Both models detected most fraudulent transactions in the reported test results.
- SMOTE increased fraud recall from approximately 81.63% to 83.67%.
- The SMOTE model detected two additional fraudulent transactions out of 98.
- False positives increased from 5 to 15, meaning more legitimate transactions were incorrectly flagged.
- Precision and F1-score decreased after SMOTE, while ROC-AUC increased.
- These results illustrate that improving fraud detection requires balancing missed fraud against unnecessary alerts.

**Important:** The notebook scales `Time` and `Amount` before splitting the dataset, allowing information from the test set to influence preprocessing. Fit the scaler on training data only, transform the test data separately, rerun the experiment, and replace these reported metrics with the corrected results before drawing final conclusions.

## 📈 Visualizations

The notebook generates the following plots and analyses:

- Class distribution of legitimate and fraudulent transactions
- Transaction amount distribution
- Transaction time distribution
- Confusion matrices
- ROC curve
- Performance comparison between both models
- Class distribution before and after SMOTE
- Top 10 features ranked by Random Forest feature importance

The feature importance analysis reported `V14`, `V10`, `V4`, `V12`, and `V17` among the highest-ranked features in the SMOTE-trained Random Forest model.

Feature importance indicates how the fitted model uses features for its predictions. It does not establish that a feature causes fraud.

## 🚀 Running the Project

### Option 1: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Open the project notebook.
3. Upload or connect your dataset to Google Drive.
4. Update the dataset path if necessary.
5. Run the notebook cells in order.

### Option 2: Local Python Environment

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Place `creditcard.csv` in your working directory and update the notebook's dataset-loading code to use the local file path.

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and execute its cells in order.

The Google Colab-specific `drive.mount()` code is only needed when accessing the dataset through Google Drive.

## ⚠️ Limitations

- The dataset contains anonymized features, limiting direct interpretation of individual variables.
- SMOTE does not guarantee better performance for every fraud detection problem.
- The current notebook requires a preprocessing correction to prevent data leakage.
- The experiment evaluates a single train-test split and does not establish performance on future transactions.
- Default classification thresholds may not be optimal for real-world fraud detection.
- The model is an educational prototype and has not been validated for live banking transactions.

## 🔮 Future Improvements

- Correct the preprocessing pipeline and regenerate all evaluation results.
- Add a Precision-Recall curve and Average Precision score.
- Compare additional classifiers, such as Logistic Regression and Gradient Boosting.
- Tune the classification threshold using validation data.
- Evaluate performance across multiple splits or cross-validation folds.
- Explore model interpretability techniques such as SHAP.
- Develop a simple interactive demonstration for testing sample transactions.
- Investigate inference speed and monitoring requirements for a production-style system.

## 📚 Learning Outcomes

Through this project, I explored:

- The challenges of highly imbalanced classification.
- Training and evaluating Random Forest models.
- Applying SMOTE to training data.
- Interpreting precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Comparing model performance using visualizations.
- Understanding the importance of leakage-free preprocessing and reliable evaluation.

## 👩‍💻 Author

**Mithravinda U**

B.Tech Artificial Intelligence and Data Science

GitHub: [MithravindaU](https://github.com/MithravindaU)

---

*FraudShield AI is an educational machine learning project intended for experimentation and learning. It should not be used to make real financial decisions without extensive additional validation, testing, security controls, and monitoring.*
