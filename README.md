# FraudShield AI | Credit Card Fraud Detection

A machine learning project that explores credit card fraud detection using a Random Forest classifier and SMOTE to address class imbalance.

## Overview

Fraudulent transactions are rare compared with legitimate transactions, making fraud detection a challenging classification problem. A model can achieve high overall accuracy while still missing fraudulent transactions.

**FraudShield AI** investigates this problem by comparing a Random Forest classifier before and after applying the Synthetic Minority Over-sampling Technique (SMOTE).

The project focuses on:
- Exploratory data analysis and data quality checks
- Class imbalance analysis
- Feature scaling and classification
- SMOTE-based oversampling
- Evaluation using precision, recall, F1-score and ROC-AUC
- Confusion matrix analysis
- Random Forest feature importance

> **Project scope:** This repository contains a notebook-based machine learning experiment. The reported metrics describe the evaluated models, not a production-ready banking fraud detection service.

## Project at a Glance

| Component | Details |
|---|---|
| Problem | Credit card fraud detection |
| Task | Binary classification |
| Main classifier | Random Forest |
| Imbalance technique | SMOTE |
| Evaluation | Precision, recall, F1-score, ROC-AUC and confusion matrices |
| Interpretability | Random Forest feature importance |
| Language | Python |
| Environment | Google Colab / Jupyter Notebook |

## Dataset

The project analyses credit card transaction data containing:

- Transaction time
- Anonymized numerical features (`V1`–`V28`)
- Transaction amount
- Transaction class

### Target classes

| Class | Meaning |
|---|---|
| `0` | Normal transaction |
| `1` | Fraudulent transaction |

### Class distribution

| Transaction class | Number of transactions | Percentage |
|---|---:|---:|
| Normal | 284,315 | 99.83% |
| Fraud | 492 | 0.17% |
| **Total** | **284,807** | **100%** |

The substantial class imbalance makes recall, precision and other class-sensitive evaluation metrics particularly important.

**Dataset note:** The dataset file is not included in the repository's listed files. Obtain the dataset from its original source, follow its terms of use, and configure the notebook to point to the downloaded CSV file.

## Methodology

### 1. Data loading and exploration

The notebook examines the dataset to understand its structure and quality.

Analysis includes:
- Class distribution
- Missing-value detection
- Duplicate-row detection
- Transaction amount distribution
- Transaction time distribution

The existing analysis reports **1,081 duplicate rows** and **no missing values in the analysed columns**.

### 2. Preprocessing

The workflow includes preparing the features and target, feature scaling, and splitting the data for model evaluation.

### 3. Baseline model

A Random Forest classifier is evaluated without SMOTE to establish a baseline for comparison.

### 4. Class imbalance handling

SMOTE generates synthetic minority-class examples to address the under-representation of fraudulent transactions.

For a valid evaluation, SMOTE must be applied **only to the training data after the train-test split**. The test set should retain its original class distribution. Confirm this sequence in the notebook before treating the reported results as a final evaluation.

### 5. Model evaluation

The models are compared using:
- **Precision:** How many transactions predicted as fraudulent were actually fraudulent?
- **Recall:** How many actual fraudulent transactions did the model identify?
- **F1-score:** A balance between precision and recall.
- **ROC-AUC:** How well the model separates the two classes across classification thresholds.
- **Confusion matrix:** The counts of correct and incorrect predictions.

### 6. Feature importance

Random Forest feature importance is used to identify which anonymized input features contributed most to the model's predictions.

## Workflow

```text
Credit Card Transaction Dataset
              |
              v
     Data Loading & EDA
              |
              v
       Data Quality Checks
              |
              v
     Feature Preparation
              |
              v
        Train-Test Split
              |
              v
    +---------------------+
    | Random Forest       |
    | Baseline Model      |
    +---------------------+
              |
              v
     Baseline Evaluation
              |
              v
   SMOTE on Training Data*
              |
              v
    +---------------------+
    | Random Forest       |
    | with SMOTE          |
    +---------------------+
              |
              v
    Compare Model Results
              |
              v
    Feature Importance
```

`*` Confirm that the notebook applies SMOTE to the training partition only.

## Results

The repository reports the following model comparison:

| Metric | Random Forest without SMOTE | Random Forest with SMOTE |
|---|---:|---:|
| Precision | 0.941 | 0.845 |
| Recall | 0.816 | 0.837 |
| F1-score | 0.874 | 0.841 |
| ROC-AUC | 0.963 | 0.973 |

These are the values reported in the existing project README. They should be treated as reported results until the notebook's evaluation procedure has been verified.

### Confusion matrices

The matrices below use **actual classes as rows** and **predicted classes as columns**.

**Random Forest without SMOTE**

| Actual class | Predicted normal | Predicted fraud |
|---|---:|---:|
| Normal | 56,859 | 5 |
| Fraud | 18 | 80 |

**Random Forest with SMOTE**

| Actual class | Predicted normal | Predicted fraud |
|---|---:|---:|
| Normal | 56,849 | 15 |
| Fraud | 16 | 82 |

The test-set confusion matrices contain 98 actual fraudulent transactions in each evaluation.

### What the results suggest

- **Fraud recall increased:** the reported value rose from approximately 81.63% to 83.67%.
- **ROC-AUC improved:** the reported value rose from 0.963 to 0.973.
- **Precision decreased:** the SMOTE model produced more false fraud predictions.
- **F1-score decreased:** the reported balance between precision and recall was lower for the SMOTE model.

SMOTE therefore did not improve every metric. The comparison illustrates the trade-off between identifying more fraudulent transactions and avoiding false alarms.

## Feature Importance

The existing notebook results report the following top 10 Random Forest features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `V14` | 0.1956 |
| 2 | `V10` | 0.1105 |
| 3 | `V4` | 0.1061 |
| 4 | `V12` | 0.0951 |
| 5 | `V17` | 0.0850 |
| 6 | `V3` | 0.0605 |
| 7 | `V11` | 0.0564 |
| 8 | `V16` | 0.0438 |
| 9 | `V2` | 0.0370 |
| 10 | `V9` | 0.0262 |

The `V` features are anonymized numerical variables. Their importance scores indicate their relative contribution to the fitted Random Forest model; they do not establish causation or explain the original meaning of those variables.

## Visualizations

The project notebook includes analysis and visualizations covering:

- Class distribution
- Transaction amount distribution
- Transaction time distribution
- Confusion matrix heatmaps
- ROC curve
- Before-and-after SMOTE comparison
- Model performance comparison
- Top 10 feature importance

These visualizations support the model evaluation and help communicate the effects of class imbalance.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Model training and evaluation |
| Imbalanced-learn | SMOTE implementation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Google Colab / Jupyter | Notebook environment |

## Reproducibility

### Repository contents

- `Mlfrauddetection.ipynb` - Main machine learning notebook
- `README.md` - Project documentation
- PNG screenshots - Saved project visuals

### Run the notebook

**Option 1: Google Colab**

1. Open the repository and select `Mlfrauddetection.ipynb`.
2. Download the notebook or open it in Google Colab.
3. Obtain the dataset separately from its original source.
4. Upload the dataset to your runtime or connect the storage location used by the notebook.
5. Update the dataset path in the notebook to match your file location.
6. Run the cells in order and review the evaluation outputs.

**Option 2: Local Jupyter environment**

1. Clone the repository:

   ```bash
   git clone https://github.com/MithravindaU/FraudShield-AI-.git
   cd FraudShield-AI-
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```powershell
   .venv\Scripts\activate
   ```

   macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the required libraries:

   ```bash
   python -m pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Mlfrauddetection.ipynb`, configure the dataset path, and execute the notebook cells in order.

**Reproducibility note:** The commands above install the main libraries used by the project. Exact package versions, the original dataset download URL, and the notebook's file path should be documented or pinned once verified against the working environment.

## Key Learnings

This project explores:

- Building a binary classification model with Random Forest
- Working with highly imbalanced transaction data
- Applying SMOTE to address minority-class under-representation
- Interpreting precision, recall, F1-score and ROC-AUC
- Reading confusion matrices to understand model errors
- Comparing baseline and resampled model performance
- Interpreting feature importance
- Understanding why accuracy alone is insufficient for fraud detection

## Limitations

- The results depend on the dataset, train-test split and model configuration.
- SMOTE can change the balance between fraud detection and false positives.
- Anonymized features limit direct interpretation of the underlying transaction behaviour.
- Feature importance does not establish causation.
- The repository does not currently document a verified production deployment or real-time transaction pipeline.
- The reported metrics should be rechecked after confirming the preprocessing sequence and ensuring that no information leaks from the test set into training.

## Future Improvements

Potential extensions include:

- Verify the evaluation pipeline and prevent data leakage
- Add stratified cross-validation
- Tune model hyperparameters
- Compare precision-recall curves and optimize the decision threshold
- Compare Random Forest with Logistic Regression and other suitable classifiers
- Add explainability with SHAP
- Build an interactive results dashboard
- Explore a carefully evaluated transaction-scoring interface

These are proposed improvements, not features claimed to be implemented.

## Author

**Mithravinda U**

- GitHub: [@MithravindaU](https://github.com/MithravindaU)
- Repository: [FraudShield AI](https://github.com/MithravindaU/FraudShield-AI-)

---

*FraudShield AI is an educational machine learning project. It is not a certified financial security product and should not be used to make real-world banking decisions without extensive validation, security review and monitoring.*
