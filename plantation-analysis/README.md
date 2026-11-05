# Banana Plantation Analysis and Regression Model Building 

## Overview
This project explores banana plantation production data to identify patterna and factors associated with banana yield and growth.

Using a dataset containing approximately **16,787 observations and 15 variables**, the analysis examines production weight, harvest timing, injection timing, caliper measurements, plantation phases, and generations.

The objective is to better understand agricultural production patterns and generate insights that could support plantation monitoring and decision-making.

## Objectives
- Finding the plantation phase that produces the highest average gross yield
- Finding how many weeks after injection are associated with the highest average caliper and gross yield
- Finding the average gross yield at 12 weeks after injection
- Finding the presence of a relationship between rainfall and banana harvest
- Finding how production patterns differ across years

## Dataset
- Source: Private banana plantation (non-public operational dataset)
- Number of observations: 16,787
- Main variables: Harvest week, Injection week, Caliper measurement, Gross production weight in kilograms, Plantation production phase, Plantation generation, Number of weeks after injection, Gross weight per banana hand
- Target variable: Gross weight per banana hand

## Tools and Technologies
R / R Markdown / Data Visualization / Statistical Analysis / Python

## Methodology
1. Data collection and understanding
2. Data preprocessing
3. Exploratory data analysis
4. Statistical analysis or modeling
5. Predictive modeling

## Key Findings
- Highest-yielding phase: 3
- Optimal weeks after injection: 13
- Average yield at week 12: 22.54 kg
- Rainfall relationship: Considerable rainfall fluctuations, including periods of relatively low rainfall and occasional sharp increases. The third group (2024) exhibited the highest observed rainfall peak, while the first group (2022) contained observations primarily from later harvest weeks.
- Yearly differences: The average yield has decreased steadily during the years.

## Results
The chosen model produced the following evaluation metrics:
- MAE: 0.0026
- MSE: 0.0018
- RMSE: 0.0424
- R2: 0.9935

## Limitations
- Observational patterns do not necessarily establish causal relationships.
- Data quality and completeness may affect the reliability of results.
- Environmental and operational factors not captured in the dataset may influence production.
- Predictive models require appropriate validation before operational use.

## My Contributions
- Plotting findings according to the objectives above
- Creating a machine learning model to predict gross weight per banana hand
