# R Package Requirements

## Overview

This project was developed using **R and R Markdown** to estimate tsunami occurrence probabilities based on earthquake characteristics through Bayesian Logistic Regression.

The analysis includes Bayesian statistical modeling, Markov Chain Monte Carlo (MCMC) sampling, posterior inference, convergence diagnostics, posterior predictive checks, and model evaluation.

The dependencies below are based on the project's `AOL Bayes.Rmd` source code.

## Required Packages

| Package | Purpose |
|---|---|
| `rmarkdown` | Render R Markdown documents into HTML, PDF, or other formats |
| `knitr` | Execute and format R code chunks |
| `tidyverse` | Data manipulation, preprocessing, and visualization |
| `brms` | Bayesian Logistic Regression using the Stan backend |
| `bayesplot` | Visualize posterior distributions, MCMC diagnostics, and model results |
| `loo` | Leave-One-Out Cross-Validation and WAIC |
| `coda` | MCMC diagnostics and posterior sample analysis |
| `gridExtra` | Arrange multiple visualizations |
| `readr` | Import and process CSV files; included in `tidyverse` |

## Installation

Install the required packages in R or RStudio:

```r
install.packages(c(
  "rmarkdown",
  "knitr",
  "tidyverse",
  "brms",
  "bayesplot",
  "loo",
  "coda",
  "gridExtra"
))
```

The `readr` package is included in `tidyverse` and does not require separate installation.

## Optional Dependencies

The following packages are used for additional analyses.

| Package | Purpose |
|---|---|
| `pROC` | ROC curve visualization and AUC calculation |
| `rjags` | Optional JAGS-based Bayesian analysis and DIC calculation |

Install them using:

```r
install.packages(c(
  "pROC",
  "rjags"
))
```

**Note:** `rjags` requires JAGS to be installed separately. It is only necessary when running the optional JAGS-based analysis.

## System Requirements

- R
- RStudio (recommended)
- A working C++ compiler for Stan-based Bayesian modeling
- JAGS (only for optional `rjags` functionality)

The `brms` package uses Stan for Bayesian inference. Depending on the operating system, additional compiler configuration may be required.

## Running the Project

### 1. Install Dependencies

Install the required R packages using the installation commands above.

### 2. Prepare the Dataset

The R Markdown analysis references the following dataset:

`earthquake_data_tsunami.csv`

Ensure that the dataset is accessible from the working directory or update the file path in the R Markdown code.

The dataset can be obtained from:

[Global Earthquake & Tsunami Risk Assessment Dataset — Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset)

### 3. Execute the Analysis

Open `AOL Bayes.Rmd` in RStudio and select **Knit**.

Alternatively, run:

```r
rmarkdown::render("AOL Bayes.Rmd")
```

### 4. Model Artifacts

Some sections of the analysis reference previously saved `.rds` model files.

If these files are unavailable, the corresponding model-fitting steps must be executed before running sections that load the saved models.

Ensure that all file paths match the local project directory.

## Reproducibility

For reproducible results, it is recommended to:

- Use a consistent R version.
- Record package versions.
- Set random seeds where applicable.
- Use consistent preprocessing procedures.
- Document the dataset version.
- Preserve the model configuration and MCMC settings.

For more reliable dependency management, the `renv` package can be used to generate an `renv.lock` file.

Install and initialize `renv`:

```r
install.packages("renv")
renv::init()
renv::snapshot()
```

Other users can then restore the recorded package environment using:

```r
renv::restore()
```

## Notes

- This project uses **R**, so a Python `requirements.txt` is not required.
- `brms` is the primary Bayesian modeling framework.
- `pROC` supports additional classification evaluation.
- `rjags` is only needed for optional JAGS-based analysis.
- Package versions have not been pinned to a specific environment.

---

**Project:** Tsunami Occurrence Prediction Using Bayesian Logistic Regression

**Course:** Bayesian Data Analysis

**Institution:** BINUS University

**Project Type:** University Group Project
