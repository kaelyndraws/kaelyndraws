# R dependencies

`Important Graphs.Rmd` loads these packages:

- readxl
- dplyr
- ggplot2
- plotly
- lubridate
- tidyr
- stringr
- forcats
- data.table

It also calls `knitr::` functions, so `knitr` is needed to render the report. `rmarkdown` is needed to knit an `.Rmd` file to HTML.

Install with:

```r
install.packages(c(
  "readxl", "dplyr", "ggplot2", "plotly", "lubridate",
  "tidyr", "stringr", "forcats", "data.table",
  "knitr", "rmarkdown"
))
```

The report reads private Excel workbooks (`Selected Sheets 2.xlsx`, `Selected.xlsx`, and `Final Durasi Decline.xlsx`), which are not distributed in the public repository.
