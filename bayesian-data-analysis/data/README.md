# 🌊 Bayesian Data Analysis — Dataset Documentation

## 📌 Dataset Overview

This dataset was used for an academic group project in the **Bayesian Data Analysis** course at **BINUS University**.

The project investigates the relationship between earthquake characteristics and tsunami occurrence using Bayesian Logistic Regression.

The dataset contains global earthquake observations, including seismic characteristics and a binary indicator of whether a tsunami occurred.

Each observation represents an earthquake event.

## 📊 Dataset Information

| Attribute | Description |
|---|---|
| Dataset Name | Global Earthquake & Tsunami Risk Assessment Dataset |
| Dataset Source | Kaggle |
| Dataset Provider | ahmeduzaki |
| Dataset Type | Structured tabular data |
| Geographic Coverage | Global |
| Observation Unit | Individual earthquake event |
| Target Variable | `tsunami` |
| Target Type | Binary |
| Machine Learning Task | Binary Classification |
| Number of Records | Not specified in the project report |
| Public Availability | Publicly listed on Kaggle |
| Original Project Purpose | Academic Bayesian statistical analysis |

**Original Dataset:**

[Global Earthquake & Tsunami Risk Assessment Dataset — Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset)

## 🗂️ Data Dictionary

The following variables were identified in the project report.

| Variable | Description | Data Type |
|---|---|---|
| `tsunami` | Indicates whether a tsunami occurred | Binary |
| `magnitude` | Earthquake magnitude | Numerical |
| `depth` | Earthquake focal depth | Numerical |
| `dmin` | Distance to the nearest seismic station | Numerical |
| `gap` | Seismic azimuthal gap | Numerical |
| `mmi` | Modified Mercalli Intensity | Numerical |
| `cdi` | Community Internet Intensity | Numerical |
| `sig` | Seismic significance | Numerical |

**Note:** The project report describes `dmin` inconsistently as distance to a seismic station or coastline. The exact definition and measurement units should be verified against the original dataset documentation.

### Target Variable

The target variable, `tsunami`, represents whether a tsunami was associated with an earthquake event.

| Encoded Value | Description |
|---|---|
| `0` | No tsunami |
| `1` | Tsunami |

This binary encoding was used as the response variable for Bayesian Logistic Regression.

## 🧹 Data Quality and Preprocessing

The original dataset was prepared before statistical modeling.

### 1. Missing and Inconsistent Values

Observations containing missing or inconsistent values were removed during preprocessing.

The final analysis used a cleaned dataset containing the selected earthquake characteristics and the binary tsunami outcome.

The project report does not specify the exact number of observations removed.

### 2. Feature Selection

The analysis focused on earthquake-related predictors considered relevant to tsunami occurrence:

- Earthquake magnitude.
- Earthquake depth.
- Distance to the nearest seismic station (`dmin`).
- Seismic azimuthal gap.
- Modified Mercalli Intensity.
- Community Internet Intensity.
- Seismic significance.

### 3. Target Preparation

The `tsunami` variable was represented as a binary outcome suitable for Bernoulli likelihood modeling.

### 4. Model Input Preparation

The cleaned predictor variables were used to construct a Bayesian Logistic Regression model.

The report documents weakly informative priors for regression coefficients and posterior inference using Markov Chain Monte Carlo.

Further implementation details can be found in the main project documentation.

## 📈 Dataset Characteristics

The dataset includes earthquake observations with varying seismic characteristics.

The main analytical focus was to examine how these characteristics relate to the probability of tsunami occurrence.

The study considered differences in earthquake magnitude, depth, shaking intensity, and other seismic measurements.

The dataset also presents challenges relevant to probabilistic modeling, including potentially imbalanced outcomes and uncertainty in predictor relationships.

## 🔓 Data Source and Availability

**Source:** Kaggle — Global Earthquake & Tsunami Risk Assessment Dataset

**Dataset Link:** https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset

The dataset is publicly listed on Kaggle.

Users can visit the original dataset page to review its documentation, availability, and applicable usage terms.

The original dataset's license and redistribution permissions should be checked before including a copy in this repository.

## 🔁 Reproducibility

To reproduce the dataset preparation:

1. Obtain the original dataset from the authorized Kaggle source.
2. Inspect the dataset structure and variable definitions.
3. Identify missing or inconsistent observations.
4. Apply the preprocessing steps used in the project.
5. Select the relevant predictor variables and binary target.
6. Use the prepared dataset for Bayesian Logistic Regression.

Exact reproduction requires the original preprocessing code and the same dataset version.

## ⚠️ Dataset Limitations

- The dataset's representativeness of global earthquake events has not been independently established.
- The frequency of tsunami events may differ from that of non-tsunami events.
- Missing or inconsistent observations may affect the available sample.
- Some important geophysical characteristics, such as fault mechanisms and tectonic settings, may not be represented.
- Correlations among seismic predictors may complicate statistical interpretation.
- Observed relationships should not automatically be interpreted as causal effects.
- The dataset alone is insufficient to establish the suitability of a model for operational tsunami warning systems.

## 📚 Academic Context

**Course:** Bayesian Data Analysis

**Institution:** BINUS University

**Project:** Prediction of Tsunami Occurrence Probability Based on Earthquake Characteristics Using Bayesian Logistic Regression

**Project Type:** University Group Project

**Documentation Author:** Kaelyn Renata Halim

---

*This documentation describes the dataset used for Bayesian probabilistic modeling of tsunami occurrence, including its source, structure, preparation, availability, and limitations.*
