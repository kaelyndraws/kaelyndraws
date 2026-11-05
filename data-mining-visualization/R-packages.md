# R Package Requirements

## Overview

This project was developed using **R and R Markdown** for data mining, statistical analysis, and data visualization.

The following R packages should be installed before running the analysis.

## Required Packages

| Package | Purpose |
|---|---|
| `rmarkdown` | Rendering R Markdown documents |
| `knitr` | Executing and formatting R code chunks |
| `data.table` | Fast data manipulation, aggregation, and joins |
| `e1071` | Statistical and probability functions |
| `dplyr` | Data manipulation |
| `vcd` | Methods, summary and inference procedures, and specialized graphics for categorical data |
| `plotly` | Interactive, web-based data visualization |
| `ggplot2` | Data visualization |

## Installation

Install the required packages in R:

```r
install.packages(c(
  "rmarkdown",
  "knitr",
  "data.table",
  "e1071",
  "dplyr",
  "vcd",
  "plotly",
  "ggplot2"
))
```

## Running the Project

Open the `.Rmd` file in RStudio and select **Knit**, or execute:

```r
rmarkdown::render("FinalExam-2802420731-Kaelyn Renata Halim.Rmd")
```

## Reproducibility

Package versions may affect the results of the analysis. For more reproducible dependency management, consider using the `renv` package to generate an `renv.lock` file.
