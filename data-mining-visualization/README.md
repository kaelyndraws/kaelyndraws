# Data Mining & Visualization Course Final Project: Marketing Campaign

## Overview
This project was developed as part of the **Data Mining and Visualization** coursework at BINUS University.

The project focuses on applying data mining and visualization techniques to explore structured datasets, identify meaningful patterns, and communicate analytical findings.

The primary objective is to transform raw data into interpretable information through systematic data preparation, exploratory analysis, and visual representation.

## Objectives
- Understand the structure and characteristics of the selected dataset.
- Prepare and preprocess data for analysis.
- Explore relationships, distributions, and patterns within the data.
- Apply appropriate data mining techniques.
- Develop visualizations that communicate analytical findings.
- Interpret the results and discuss their potential implications.

## Dataset
- Source: BINUS University
- Number of observations: 8,238
- Main variables: Age of the client, Type of job, Marital status, Highest education level attained, Whether the client has a credit in default, Whether the client has a housing loan, Whether the client has a personal loan, Type of communication used, Month of the last contact, Day of the week when last contact occurred, Duration of the last contact in seconds, Number of contacts made during the current campaign, Days since the last contact in a previous campaign, Number of previous contacts before the current campaign, Outcome of the previous campaign
- Target variable: Whether the client subscribed to a term deposit

## Tools and Technologies
R / dplyr / ggplot2 / RStudio

## Methodology
1. Data collection and understanding
2. Data preprocessing
3. Exploratory data analysis
4. Data mining
5. Data visualization

## Key Findings
- The following pairs of numerical variables are meaningfully correlated:
  - Age of the client and Number of previous contacts before the current campaign
  - Duration of the last contact in seconds and Number of contacts made during the current campaign
  - Duration of the last contact in seconds and Number of previous contacts before the current campaign
  - Number of contacts made during the current campaign and Number of previous contacts before the current campaign
- The following categorical variables have a relationship with Whether the client subscribed to a term deposit:
  - Type of job
  - Marital status
  - Highest education level attained
  - Whether the client has a credit in default
  - Whether the client has a housing loan
  - Type of communication used
  - Month of the last contact
  - Outcome of the previous campaign
- Potential influencers of the campaign outcome:
  - Call duration: Clients who responded “yes” consistently had much longer average call times, indicating higher engagement. It shows a strong positive association with campaign success.
  - Contact method: Success rates varied depending on the communication channel used, with cellular contact performing better.
  - Previous campaign outcome: Clients who had a positive outcome in the prior campaign were more likely to respond positively again. It shows that the outcome of the previous campaign is closely linked to current success.

## Results
The median age of clients is 38 years old, and those who responded positively to the campaign (y = “yes”) had significantly longer call durations, with a median of 452.5 seconds (or around 7.542 minutes), compared to those who declined. The highest positive response rates were observed among certain job types like students and retirees, despite their smaller representation in the dataset.

An unusual pattern was seen in the <30 age group, which had lower response rates despite consistent outreach.

While variables like duration and previous show correlation, these may not reflect causality - longer calls could be a result of interest rather than a cause of success.

Business recommendations:
- Prioritize clients with previous successful contact or longer call durations: Allocate more effort to clients with a history of previous contacts and ensure call durations exceed 7 minutes (or 420 seconds), as these patterns are linked with higher campaign success.
- Target students and retired clients more actively: Future campaigns should focus on students and retirees, as these groups had the highest success rates despite being smaller segments.

## Limitations
- Findings depend on the quality and representativeness of the dataset.
- Missing values and preprocessing decisions may influence the results.
- Observed correlations do not necessarily imply causation.
- Data mining methods may be sensitive to parameter choices and feature selection.
- Additional validation may be necessary before applying the findings in real-world settings.
