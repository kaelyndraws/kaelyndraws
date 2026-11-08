# 📊 Survey and Sampling Methods — Dataset Documentation

## 📌 Dataset Overview

This dataset was collected for an academic group research project in the **Survey and Sampling Methods** course at **BINUS University**.

The study, titled *"Survei Pengaruh Kefasihan Bahasa Inggris terhadap Nilai Akhir Delivery in English"*, investigates the relationship between English proficiency and academic performance among BINUS Data Science B28 students.

Primary data was collected through a structured online questionnaire examining students' English-language habits, self-assessed proficiency, grammar competency, and academic grades.

## 📊 Dataset Information

| Attribute | Description |
|---|---|
| Dataset Name | English Proficiency and Academic Performance Survey |
| Dataset Source | Primary survey data |
| Data Collection Platform | Google Forms |
| Data Collection Period | May 21 – June 2, 2026 |
| Target Population | BINUS Data Science B28 students |
| Population Size | 140 students |
| Sampling Method | Convenience Sampling |
| Minimum Target Sample | 60 respondents |
| Original Responses Collected | 70 |
| Processed Dataset | 69 observations |
| Number of Columns | 51 |
| Missing Values | 0 |
| Exact Duplicate Rows | 0 |
| Dataset Format | CSV |
| File Name | `DataAfter_11.csv` |
| Original Purpose | Academic research |

**Note:** The research report records 70 collected responses, while the provided processed CSV contains 69 observations. The exact reason for this difference should be verified against the preprocessing code.

## 🗂️ Data Dictionary

The processed dataset contains original questionnaire responses and additional variables generated during preprocessing.

### 1. Survey Information and Academic Background

| Variable / Field | Description |
|---|---|
| `Timestamp` | Timestamp associated with survey submission |
| Delivery in English Participation | Whether the respondent has taken or is taking a Delivery in English course |
| Beelingua Journey | English course level completed by the respondent |
| External English Training | Whether the respondent has attended English training outside the university |
| Course Grades | Self-reported academic grades across four courses |

The four academic courses included:

- Data Mining and Visualization
- Data Structures
- Data Management and Organization
- Machine Learning

### 2. English Language Habits and Self-Assessment

The questionnaire includes Likert-scale responses measuring:

- Consumption of English-language media without Indonesian subtitles.
- Confidence in presenting academic work in English.
- Ability to read English technical documentation.
- Self-assessed speaking, listening, reading, and writing abilities.
- Difficulties understanding English-language academic instructions.
- Challenges interpreting English vocabulary and technical explanations.
- Perceived academic benefits of bilingual instruction.

These variables provide additional context for understanding students' English-language experiences.

### 3. English Competency Assessment

The dataset includes responses to 11 English competency questions.

The questions assess grammar, vocabulary, reading comprehension, and appropriate language usage.

Each question was scored using binary encoding:

| Score | Meaning |
|---|---|
| `1` | Correct answer |
| `0` | Incorrect answer |

The resulting columns are:

`Q1_score` through `Q11_score`

### 4. Engineered Variables

| Variable | Description | Data Type |
|---|---|---|
| `Y_Nilai_Matkul` | Average numerical grade across four academic courses | Float |
| `X1_Skor_Tes` | Total correct answers on the 11-question English competency assessment | Integer |
| `X2_Level_Beelingua` | Numerically encoded Beelingua Journey level | Integer |
| `X3_Kebiasaan` | Aggregated score from three English-language habit questions | Integer |

The processed CSV also includes four numerical grade columns with the suffix `_num`, representing converted course grades.

## 🧹 Data Quality and Preprocessing

### 1. Respondent Screening

Responses were screened based on eligibility for the target population.

The study focused on BINUS Data Science B28 students who had taken the relevant Delivery in English courses.

The supplied processed dataset contains 69 observations.

### 2. Missing Value Assessment

All questionnaire questions were configured as mandatory.

An inspection of the processed CSV confirmed:

- Missing values: **0**
- Exact duplicate rows: **0**

These results describe the supplied processed dataset and do not independently establish the completeness of the original survey responses.

### 3. Academic Grade Encoding

Letter grades were converted into numerical values.

| Letter Grade | Numerical Value |
|---|---:|
| A | 4.00 |
| A- | 3.67 |
| B+ | 3.33 |
| B | 3.00 |
| B- | 2.50 |
| C | 2.00 |
| D | 1.00 |

The four converted course grades were averaged to construct the dependent variable, `Y_Nilai_Matkul`.

### 4. English Competency Scoring

The 11 English competency questions were evaluated using binary scoring.

Correct answers received one point, while incorrect answers received zero points.

The scores were summed to create `X1_Skor_Tes`, with a theoretical range of 0–11.

### 5. Beelingua Level Encoding

Categorical Beelingua Journey levels were converted into numerical values.

| Beelingua Journey | Encoded Value |
|---|---:|
| Rookie | 1 |
| Explorer | 2 |
| Maverick | 3 |

The encoded variable was stored as `X2_Level_Beelingua`.

### 6. English Language Habits

Three Likert-scale questions were aggregated to construct `X3_Kebiasaan`.

This variable represents the combined score of selected English-language habits and confidence-related responses.

## 📈 Dataset Characteristics

The processed dataset was used to examine relationships between English proficiency indicators and academic performance.

The main analysis focused on:

- English competency scores.
- Beelingua course levels.
- English-language habits.
- Average academic grades.

The dataset supports descriptive analysis, correlation analysis, questionnaire reliability testing, and multiple linear regression.

## 🔬 Measurement Quality

The study evaluated the internal consistency of selected questionnaire components using Cronbach's Alpha.

| Component | Cronbach's Alpha |
|---|---:|
| English Competency Assessment | 0.736 |
| English Language Habits | 0.706 |

Both values exceeded the study's selected threshold of 0.60.

The report also documents questionnaire validity testing using Pearson correlation.

## 🔒 Data Privacy and Availability

**Source:** Primary survey responses collected from BINUS University students.

**Access Status:** Academic research dataset; public redistribution permission has not been established.

The survey was conducted anonymously, with voluntary participation and confidential treatment of responses.

Although the dataset does not necessarily contain direct personal identifiers, combinations of academic grades, course levels, survey answers, and timestamps may create reidentification risks.

For this reason, the original respondent-level dataset should not be published without confirming that public sharing is permitted and that appropriate privacy safeguards have been applied.

The repository may instead provide:

- Dataset documentation.
- Survey methodology.
- Preprocessing code, where publication is permitted.
- Approved aggregate statistics.
- Approved visualizations and analytical findings.

## 🔁 Reproducibility

To reproduce the dataset preparation and analysis:

1. Obtain authorized access to the original survey responses.
2. Apply respondent eligibility screening.
3. Verify missing values and duplicate records.
4. Convert academic letter grades into numerical values.
5. Score the 11 English competency questions.
6. Encode Beelingua Journey levels.
7. Aggregate selected Likert-scale responses.
8. Construct the final dependent and independent variables.
9. Perform the statistical analyses documented in the main project repository.

Exact reproduction requires the original survey data, questionnaire scoring rules, and preprocessing code.

## ⚠️ Dataset Limitations

- **Convenience Sampling:** The respondents were not selected through probability-based sampling, limiting population-level generalizability.
- **Limited Population:** The study focuses on one university cohort.
- **Self-Reported Information:** Academic grades and language habits may be affected by reporting or recall bias.
- **Measurement Bias:** Self-assessment questions may not perfectly represent actual English proficiency.
- **Sample Size:** The processed dataset contains 69 observations, limiting the precision of some statistical estimates.
- **Ordinal Encoding:** Beelingua levels were represented numerically, which introduces assumptions when used in linear regression.
- **Observational Data:** Statistical relationships cannot establish causation.
- **Privacy Restrictions:** Respondent-level records may not be suitable for unrestricted public distribution.

## 📚 Academic Context

**Course:** Survey and Sampling Methods

**Institution:** BINUS University

**Project Type:** University Group Project

**Research Title:** Survei Pengaruh Kefasihan Bahasa Inggris terhadap Nilai Akhir "Delivery in English"

**Research Team:**
- Kaelyn Renata Halim
- Sherly Davina Winarlie
- Solascriptura Bintang Sulaiman

---

*This dataset documentation supports an academic study of the relationships between English proficiency, learning habits, and academic performance using primary survey data.*
