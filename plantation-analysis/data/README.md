# Dataset Documentation — Banana Plantation Analysis

## 1. Dataset Overview

This dataset contains agricultural production records obtained from a privately operated banana plantation in Indonesia.

The dataset was used to investigate banana production patterns, harvest timing, plantation characteristics, and potential relationships between environmental factors and production yield.

| Attribute | Description |
|---|---|
| Dataset Name | Banana Plantation Production Dataset |
| Source | Private banana plantation |
| Dataset Type | Structured agricultural data |
| Number of Records | Approximately 16,787 |
| Number of Variables | 15 |
| Geographic Coverage | Indonesia |
| Data Availability | Private / Restricted |
| Primary Analysis Tools | R, R Markdown |

## 2. Data Dictionary

The following variables were used in the analysis.

| Variable | Description | Unit / Type |
|---|---|---|
| Week Panen | Week in which harvesting occurred | Week number |
| Week Inject | Week in which injection occurred | Week number |
| Kaliper | Caliper measurement of the banana crop | Millimeters |
| Bruto (Kg) | Gross production weight | Kilograms |
| Phase | Plantation production phase | Categorical |
| Generasi | Plantation generation | Categorical |
| Week after Inject | Number of weeks elapsed after injection | Weeks |
| Bruto/Sisir | Gross production weight per banana hand | Kilograms |
| Curah Hujan | Rainfall measurement associated with the plantation observations | Millimeters |

**Note:** This table documents the variables identified in the project. The remaining variables and exact measurement definitions should be verified against the original dataset.

## 3. Data Characteristics

The dataset includes observations related to:

- Banana harvest and production timing.
- Gross banana production weight.
- Crop growth characteristics.
- Plantation phases and generations.
- Rainfall patterns across harvest weeks.

The analysis includes comparisons across production conditions and time periods.

## 4. Data Preparation

The dataset was prepared for exploratory data analysis and statistical investigation.

Relevant preparation activities may include:

- Inspecting variable types and dataset structure.
- Checking missing or inconsistent observations.
- Grouping observations by harvest week, phase, or generation.
- Calculating summary statistics.
- Preparing data for visualizations and comparisons.

Specific preprocessing steps are documented in the project's analysis files.

## 5. Data Availability and Confidentiality

**The original dataset is not publicly available.**

The data originates from a private banana plantation and may contain confidential operational information.

To protect the data provider's confidentiality:

- The original dataset is not distributed through this repository.
- Identifying or sensitive operational information is not disclosed.
- Any published analytical results or visualizations must be approved for public sharing where required.

The repository is intended to demonstrate analytical methodology and technical skills without exposing confidential source data.

## 6. Reproducibility

Because the original dataset is restricted, external users may not be able to reproduce the analysis directly.

The repository provides analytical code and methodological documentation where publication is permitted.

Reproduction of the original findings requires authorized access to the underlying dataset.

## 7. Dataset Limitations

- The dataset represents observations from a specific private plantation.
- Results may not generalize to other plantations or agricultural environments.
- Data quality and measurement practices may influence analytical outcomes.
- Observed associations should not automatically be interpreted as causal relationships.

---

**Data Source:** Private banana plantation, Indonesia

**Project Author:** Kaelyn Renata Halim

**Purpose:** Educational and personal data analysis portfolio
