# Survey and Sampling Methods – English Proficiency & Academic Performance

## Overview
This project investigates the relationship between English proficiency and academic performance among Data Science students at BINUS University.

Developed as part of the Survey and Sampling Methods coursework, the study examines whether English grammar proficiency, English course levels, and language-learning habits are associated with students' academic performance in courses delivered in English.

The research combines primary data collection, questionnaire design, statistical validation, reliability testing, data preprocessing, and multiple linear regression to identify meaningful relationships between English proficiency and academic outcomes.

The study focuses on **BINUS Data Science B28 students** who have completed courses delivered in English.

## Objectives
- Design a structured questionnaire to measure students' English proficiency and language-learning habits.
- Determine an appropriate sample size for the target population.
- Collect primary data through an online survey.
- Evaluate questionnaire validity and reliability.
- Prepare survey responses for statistical analysis.
- Examine correlations between English proficiency indicators and academic performance.
- Develop a multiple linear regression model to identify statistically significant predictors of academic grades.

## Dataset
- Source: Primary survey data
- Number of observations: 70
- Main variables: Course grade, Mini test score, Beelingua level, English habits

## Tools and Technologies
Google Forms / Python / Sample Size Estimation / Finite Population Correction / Filtering / Encoding / Feature Aggregation / Pearson Correlation / Multiple Linear Regression / Cronbach's Alpha / Correlation Heatmap / Descriptive Statistics

## Methodology
1. Sampling Design
2. Questionnaire Design
3. Data Collection
4. Data Preprocessing
5. Measurement Quality Assessment
6. Correlation Analysis
7. Multiple Linear Regression

## Key Findings
- **Grammar proficiency** was the only statistically significant predictor of academic performance in the fitted regression model.
- Each additional point in the grammar test was associated with approximately a 0.10-point increase in predicted academic grade, holding the other predictors constant.
- Beelingua course level and English-language habits showed positive estimated coefficients but were not statistically significant after accounting for the other variables.
- The findings suggest that objectively measured grammar competency may be a more informative indicator of academic performance than self-reported habits within this sample.

These findings describe statistical associations and do not establish causal effects.

## Results
1. Correlation Analysis
   
   All three independent variables showed positive correlations with academic performance.
   | Variable | Pearson Correlation with Academic Grade |
   | --- | --- |
   | Grammar Test Score | 0.57 |
   | Beelingua Course Level | 0.48 |
   | English Language Habits | 0.39 |

   The grammar competency score demonstrated the strongest positive correlation with academic grades.
2. Regression Model Performance
   | Metric | Result |
   | --- | --- |
   | R-squared | 0.345 |
   | Adjusted R-squared | 0.314 |
   | F-statistic | 11.39 |
   | Model p-value | Approximately 4.26 * 10^-6 |
   | Regression Observations | 69 |

   The regression model explained approximately **34.5% of the observed variation in academic grades**.

  The remaining variation was not explained by the three predictors included in the model.
3. Regression Coefficients
| Predictor | Coefficient | p-value | Statistically Significant (α = 0.05) |
| --- | --- | --- | --- |
| Grammar Test Score | 0.1001 | 0.004 | Yes |
| Beelingua Course Level | 0.0802 | 0.318 | No |
| English Language Habits | 0.0077 | 0.709 | No |

## Limitations
- **Convenience Sampling**: Participants were recruited through accessible student networks rather than probability-based sampling.
- **Limited Population**: The research focused on one university cohort.
- **Self-Reported Responses**: Language habits and some academic information may be affected by recall or response bias.
- **Model Explanatory Power**: Approximately 65.5% of grade variation remained unexplained.
- **Cross-Sectional Design**: The analysis cannot establish causal relationships.
- **Ordinal Encoding**: Treating Beelingua levels as equally spaced numerical values introduces a modeling assumption.
- **Sample Count Difference**: Although 70 responses were collected, the regression output reported 69 observations. The reason should be verified against the analysis code.

## My Contributions
- Proposed the research topic.
- Contributed to questionnaire design.
- Contributed to coding and implementation.
