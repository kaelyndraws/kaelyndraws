# 💳 Credit Score Classification — Dataset Documentation

## 📌 Dataset Overview

This dataset was provided by **BINUS University** through Google Drive as part of an academic examination in the Data Science program.

The dataset contains financial, demographic, and credit-related information used to develop and evaluate machine learning models for predicting credit score categories.

The primary objective is to classify observations into three credit score classes based on the available features.

## 📊 Dataset Information

| Attribute | Description |
|---|---|
| Dataset Name | Credit Score Classification Dataset |
| Source | BINUS University |
| Distribution Method | University-provided Google Drive link |
| Original Purpose | Academic examination |
| Dataset Type | Structured tabular data |
| Number of Records | Approximately 25,000 |
| Target Variable | `Credit_Score` |
| Number of Target Classes | 3 |
| Target Encoding | 0, 1, 2 |
| Machine Learning Task | Multiclass Classification |
| Public Availability | Not confirmed |

## 🗂️ Data Description

The dataset consists of features describing individuals' financial characteristics, credit histories, and borrowing behavior.

These features are used to predict the target variable, `Credit_Score`.

### Target Variable

| Encoded Class | Description |
|---|---|
| 0 | Credit score category 0 |
| 1 | Credit score category 1 |
| 2 | Credit score category 2 |

**Note:** The original category names and their mapping to the encoded values should be verified against the dataset's preprocessing code.

### Feature Information

The dataset contains numerical and categorical features relevant to credit score classification.

One categorical feature, `Type_of_Loan`, contains information about loan types and requires additional preprocessing to convert its values into machine-readable features.

The complete feature list and definitions can be obtained from the original dataset, subject to university permissions.

## 🧹 Data Quality and Preprocessing

During the project, several data quality issues were investigated and addressed.

### 1. Missing Values

Missing values were handled according to the proportion of missing observations:

| Missing Percentage | Preprocessing Strategy |
|---|---|
| Below 5% | Remove affected rows |
| 5%–40% | Apply imputation |
| Above 40% | Remove affected columns |

These thresholds reflect the preprocessing approach used in the academic project.

### 2. Invalid Values and Outliers

The dataset contained values requiring further investigation, including:

- Unrealistic age values.
- Unusually high interest rates.
- Negative values in variables where they may be invalid.
- Inconsistent categorical entries.

These observations were examined during data preprocessing.

### 3. Categorical Feature Encoding

Categorical features were transformed into numerical representations suitable for machine learning.

For example, `Type_of_Loan` required additional processing to separate and encode loan-type information.

### 4. Feature Preparation

The processed dataset was prepared for classification model development and evaluation.

Preprocessing included feature transformation and preparation of model input variables.

## 🎯 Target Distribution

The held-out test dataset contained 5,000 observations distributed across the three encoded classes.

| Class | Test Observations | Percentage |
|---|---:|---:|
| 0 | 909 | 18.18% |
| 1 | 1,440 | 28.80% |
| 2 | 2,651 | 53.02% |
| **Total** | **5,000** | **100%** |

The distribution indicates class imbalance in the test dataset, with class 2 representing the largest proportion.

These counts describe the test set rather than the complete original dataset.

## 🔒 Data Availability and Usage Restrictions

The dataset was distributed by BINUS University through a Google Drive link for academic examination purposes.

**The original dataset is not included in this public repository because permission for public redistribution has not been confirmed.**

The dataset may be subject to university access restrictions and academic-use policies.

This repository documents the dataset characteristics and preprocessing methodology without distributing the original examination data.

Users wishing to reproduce the analysis would need authorized access to the dataset.

## ⚠️ Dataset Limitations

- The dataset was provided for academic purposes and may not represent real-world credit applicants.
- Missing values, invalid entries, and outliers may influence model performance.
- Class imbalance may affect classification performance across credit score categories.
- The original data collection methodology and population representativeness have not been independently verified.
- Models developed using this dataset should not be assumed suitable for real-world credit approval decisions.

## 🔁 Reproducibility

The main project repository documents the machine learning workflow, including preprocessing, feature engineering, model development, and evaluation.

Reproducing the original results requires access to the authorized dataset and the corresponding project code.

---

**Dataset Provider:** BINUS University

**Project:** Credit Score Classification Using Machine Learning

**Project Type:** University Academic Examination

**Documentation Author:** Kaelyn Renata Halim
