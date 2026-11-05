# Credit Score Classification

## Overview
This project focuses on developing a machine learning system to classify individuals into credit score categories based on their financial and credit-related information.

Using a dataset containing approximately 25,000 observations, the project explores data preprocessing, feature engineering, classification algorithms, hyperparameter tuning, model evaluation, experiment tracking, and deployment.

The primary objective is to build a classification model capable of identifying credit score categories while demonstrating an end-to-end machine learning workflow.

## Objectives
- Explore and understand the characteristics of the credit score dataset.
- Identify and address data quality issues, including missing values, invalid entries, and outliers.
- Prepare numerical and categorical features for machine learning.
- Tran and compare classification models.
- Optimize model performance through hyperparameter tuning.
- Evaluate model performance using appropriate classification metrics.
- Track experiments and model artifacts using MLflow.
- Explore model deployment using FastAPI, Streamlit, and AWS SageMaker.

## Dataset
- Source: BINUS University
- Number of observations: 25,000
- Main variables: Age, Annual income, Monthly inhand salary, Number of bank accounts, Number of credit cards, Interest rate, Number of loans, Type of loan, Delay from due date, Number of delayed payment, Changed credit limit, Number of credit inquiries, Credit mix, Outstanding debt, Credit utilization ratio, Credit history age, Payment of minimum amount, Total EMI per month, Amount invested monthly, Payment behavior, Monthly balance
- Target variable: Credit score

## Tools and Technologies
Python / Pandas / Numpy / scikit-learn / Random Forest / MLflow / Accuracy / Precision / Recall / F1-score / FastAPI / Streamlit / AWS SageMaker / Jupyter Notebook

## Methodology
1. Exploratory data analysis
2. Data preprocessing
3. Model development
4. Hyperparameter tuning
5. Model evaluation
6. Experiment tracking with MLflow
7. Model deployment

## Results
- Developed a multiclass credit score classification workflow using approximately 25,000 observations.
- Implemented data preprocessing and feature engineering.
- Selected a Random Forest model with 500 estimators.
- Achieved 75% accuracy and 0.75 weighted F1-score on the held-out test dataset.
- Incorporated MLflow for experiment tracking.
- Developed inference components using FastAPI and Streamlit.
- Explored deployment using AWS SageMaker.

## Limitations
- Classification performance may vary across credit score categories.
- Dataset quality and feature availability can influence model predictions. Class imbalance may affect minority-class performance.
- Additional validation is required to assess generalization to new populations.
- Predictive performance alone does not establish fairness or suitability for real-world credit decisions.
