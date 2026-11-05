# 📊 Data Mining & Visualization — Dataset Documentation

## 📌 Dataset Overview

This dataset was used for an academic project in the **Data Mining and Visualization** course at **BINUS University**.

The dataset serves as the foundation for exploratory data analysis, data preprocessing, pattern discovery, and visualization.

The analysis aims to identify meaningful patterns, relationships, and trends within the data and communicate the findings through appropriate analytical and visualization techniques.

## 📊 Dataset Information

| Attribute | Description |
|---|---|
| Dataset Name | Marketing Campaign B |
| Dataset Source | BINUS University |
| Dataset Type | Structured |
| File Format | CSV |
| Number of Records | 8,238 |
| Number of Variables | 17 |
| Target Variable | Whether the client subscribed to a term deposit |
| Public Availability | Private |
| Original Purpose | Academic coursework |

## 🗂️ Data Dictionary

The data dictionary describes the variables included in the dataset, their data types, and their meanings.

| Variable | Description | Data Type | Unit / Values |
|---|---|---|---|
| age | Age of the client | Numeric | years |
| job | Type of job | Categorical | admin, blue-collar, entrpreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown |
| marital | Marital status| Categorical | divorced, married, single, unknown |
| education | Highest education level attained | Categorical | basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown |
| default | Whether the client has a credit in default | Categorical | yes, no, unknown |
| housing | Whether the client has a housing loan | Categorical | yes, no, unknown |
| loan | Whether the client has a personal loan | Categorical | yes, no, unknown |
| contact | Type of communication used | Categorical | cellular, telephone |
| month | Month of the last contact | Categorical | apr, aug, dec, jul, jun, mar, may, nov, oct, sep |
| day_of_week | Day of the week when the last contact occurred | Categorical | fri, mon, thu, tue, wed |
| duration | Duration of the last contact in seconds | Numeric | seconds |
| campaign | Number of contacts made during the current campaign | Numeric | count |
| pdays | Days since the last contact in a previous campaign | Numeric | days |
| previous | Number of previous contacts before the current campaign | Numeric | count |
| poutcome | Outcome of the previous campaign | Categorical | failure, nonexistent, success |
| y | Whether the client subscribed to a term deposit | Categorical | yes, no |
| campaign_flag | Whether the observation contains an unusually high contact count | Categorical | TRUE, FALSE |
| pdays_status | Whether the observation has been contacted | Categorical | contacted, not_contacted |
| pdays_num | Days since the last contact in a previous campaign without non-contacted | Numerical | days |

## 🧹 Data Quality and Preprocessing

The dataset was examined and prepared for data mining and visualization.

The following preprocessing activities should be documented where applicable.

### 1. Missing Values

Missing values were present in the age, job, duration, campaign, pdays, and previous variables.

**Method:** Removal, median imputation

### 2. Duplicate Records

There were no duplicate records in the dataset.

### 3. Outlier Detection

There were outliers in the age, duration, previous, campaign, and pdays variables.

**Method:** Kept, flagging

### 4. Data Transformation

**Transformations performed:** Data type conversion, feature engineering

## 📈 Data Characteristics

This section summarizes the dataset's main characteristics.

| Characteristic | Description |
|---|---|
| Numerical Variables | 5 |
| Categorical Variables | 12 |
| Missing Values | present in the age, job, duration, campaign, pdays, and previous |
| Duplicate Records | None |
| Class Distribution | Imbalanced |
| Important Relationships | age and previous, duration and campaign, duration and previous, campaign and previous, y and job, y and marital, y and education, y and default, y and housing, y and contact, y and month, y and poutcome |

## 🔒 Data Source and Availability

**Original Source:** BINUS University

**Access Status:** Private

The dataset was used for educational purposes as part of the Data Mining and Visualization course.

The availability of the original and processed datasets depends on the source's licensing terms, access restrictions, and redistribution permissions.

If redistribution is not permitted, the original dataset will not be included in this repository.

## 🔁 Reproducibility

The main project repository contains the available analytical code, visualizations, and methodological documentation.

To reproduce the analysis:

1. Obtain the dataset from its authorized source.
2. Follow the preprocessing steps documented in the project.
3. Install the required software and dependencies.
4. Execute the analysis files in the documented order.
5. Compare the generated outputs with the reported findings.

Reproducibility may be limited if the original dataset is restricted or unavailable.

## ⚠️ Dataset Limitations

- The dataset may contain missing, inconsistent, or incomplete observations.
- Data quality and preprocessing choices may influence analytical results.
- The representativeness of the dataset depends on its original collection methodology.
- Patterns identified through exploratory analysis do not necessarily imply causal relationships.
- Findings may not generalize beyond the dataset's original context.

## 📚 Academic Context

**Course:** Data Mining and Visualization

**Institution:** BINUS University

**Project Type:** University Academic Project

**Documentation Author:** Kaelyn Renata Halim

---

*This documentation describes the dataset used in the Data Mining and Visualization project, including its structure, preparation, availability, and limitations.*
