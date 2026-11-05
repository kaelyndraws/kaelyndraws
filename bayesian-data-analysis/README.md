# Tsunami Occurrence Prediction Using Bayesian Logistic Regression

## Overview
This project investigates the probability of tsunami occurrence based on earthquake characteristics using **Bayesian Logistic Regression**.

Developed as part of the **Bayesian Data Analysis** coursework at BINUS University, the study applies Bayesian statistical inference to estimate the relationship between seismic variables and tsunami occurrence while explicitly accounting for parameter uncertainty.

Unlike conventional logistic regression, which primarily provides point estimates, Bayesian Logistic Regression produces posterior distributions that allow uncertainty to be quantified and interpreted.

The project explores how probabilistic modeling may contribute to understanding tsunami risk and supporting future disaster risk assessment research.

## Objectives
- Investigate the relationship between earthquake characteristics and tsunami occurrence.
- Develop a Bayesian Logistic Regression model for binary tsunami classification.
- Incorporate weakly informative prior distributions into model estimation.
- Estimate posterior distributions using Markov Chain Monte Carlo (MCMC).
- Evaluate model convergence and parameter uncertainty.
- Assess predictive performance using posterior predictive checks, classification metrics, WAIC, and Leave-One-Out Cross-Validation (LOO).
- Interpret the importance and uncertainty of earthquake-related predictors.

## Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset) 
- Number of observations: 782
- Main variables: Earthquake magnitude, Community Decimal Intensity, Modified Mercalli Intensity, Event significance score, Number of seismic monitoring stations, Distance to nearest seismic station, Azimuthal gap between stations, Earthquake focal depth, Epicenter latitude, Epicenter longitude, Year of occurrence, Month of occurrence
- Target variable: Tsunami potential indicator

## Tools and Technologies
R / Bayesian Logistic Regression / brms / Hamiltonian Monte Carlo (HMC) / Stan / R-hat / Effective Sample Size / Trace Plots / WAIC / LOO / Posterior Predictive Checks / Accuracy / Sensitivity / Specificity / ROC-AUC

## Methodology
1. Data Preparation
2. Bayesian Logistic Regression
3. Prior Distributions
4. Posterior Inference
5. Convergence Diagnostics
6. Model Evaluation

## Key Findings
- Classification Performance
  | Metric | Result |
  | --- | --- |
  | Accuracy | 74.68% |
  | Sensitivity (Tsunami Recall) | 52.96% |
  | Specificity (Non-Tsunami Recall) | 88.49% |
  | ROC-AUC | 0.822 |
  | WAIC | 875.0 |
  | LOOIC | 875.1 |

  The model achieved a ROC-AUC of 0.822, indicating useful discrimination between tsunami and non-tsunami events in the evaluated dataset.

  However, sensitivity was approximately 52.96%, indicating that a substantial proportion of actual tsunami events were not identified at the selected classification threshold.

  This limitation is particularly important when considering potential disaster-warning applications.
- Posterior Coefficient Estimates
  | Predictor | Estimate | 95% Credible Interval |
  | --- | --- | --- |
  | Magnitude | 0.44 | [-0.02, 0.89] |
  | Depth | -0.00 | [-0.00, 0.00] |
  | dmin | 0.51 | [0.40, 0.62] |
  | Gap | 0.01 | [0.01, 0.02] |
  | MMI | -0.18 | [-0.34, -0.02] |
  | CDI | 0.16 | [0.10, 0.23] |
  Key interpretations:
  - dmin exhibited a relatively strong positive association with tsunami occurrence in the fitted model.
  - Magnitude showed a positive posterior mean, although its 95% credible interval slightly overlapped zero.
  - CDI showed a positive association with tsunami probability.
  - MMI showed a negative coefficient after accounting for other predictors.
  - Depth and seismic significance showed relatively small coefficient estimates.
  These relationships describe the fitted statistical model and should not be interpreted as causal effects.
- Posterior Predictive Checks
  Posterior predictive checks were performed to compare observed outcomes against replicated observations generated from the fitted model.

  The reported plots showed reasonable agreement between observed and replicated distributions.
- Model Fit
  The study reported:
  - WAIC: 875.0
  - LOOIC: 875.1
  - Pareto-k diagnostics below 0.7

  The close agreement between WAIC and LOOIC suggests consistency between these model evaluation estimates.

  However, these information criteria are most informative when comparing alternative models fitted to the same data.

## Results
1. Bayesian Logistic Regression provides interpretable probabilistic predictions while explicitly representing parameter uncertainty.
2. Earthquake characteristics such as dmin, magnitude, and shaking intensity showed associations with tsunami occurrence.
3. MCMC diagnostics supported satisfactory convergence.
4. The model achieved useful overall discrimination but had limited sensitivity for identifying tsunami events.
5. Further evaluation and validation would be necessary before considering real-world early-warning applications.

## Limitations
- The dataset may not represent every earthquake or tsunami-generating mechanism.
- Some relevant geophysical characteristics, such as fault mechanisms and tectonic settings, were not included.
- Correlated seismic predictors may complicate coefficient interpretation.
- The model's sensitivity may be insufficient for safety-critical applications.
- Model performance on independent future earthquake events has not been established.
- No empirical comparison with a separately fitted classical logistic regression baseline was reported.

This project is an academic statistical modeling exercise and is not an operational **tsunami warning system**.

## My Contributions
I performed model evaluation, wrote the Results & Discussion section in the report, performing final corrections to the overall report, and presented the results in the report and the presentation video.
