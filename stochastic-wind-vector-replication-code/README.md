# Stochastic Wind Vector Replication Code

Replication code and supporting data products for **"Stochastic weather generators for high-frequency wind vector time series"**.

This folder contains the final replication code for reproducing Figures 1, 12, 13, A1, B1, B2, and F1; Tables 2 and F1; and the zero-wind analysis discussed in Appendix A1. The analysis uses high-frequency wind-vector and surface meteorology data from the ARM Southern Great Plains Lamont, Oklahoma E13 site.

## Repository Contents

```text
.
|-- data/
|   |-- LamontOK_E13_19930721_20250920.csv
|   |-- training_474.csv
|   |-- testing_185.csv
|   |-- Embedded/
|   |-- Features/
|   `-- No Weather/
|-- scripts/
|   |-- 00_Combining_ARM_data.R
|   |-- 01_Figure01_Seasonality_and_Zeros.R
|   |-- 02_Imputation_and_Table2.R
|   |-- 03_No_Memorization.R
|   |-- 04_Figures12_and_13_Energy_Scores.R
|   |-- README_weather_states
|   `-- weather_states_paper.R
|-- Seasonality and Zero Analysis/
|-- Imputation Plots/
|-- Minimum Euclidean Distance/
`-- Energy Scores/
```

## Data

The ARM data cover E13 observations from `1993-07-21 00:00:00` through `2025-09-20 00:00:00`.

- `data/LamontOK_E13_19930721_20250920.csv`: combined ARM E13 data table created from daily ARM NetCDF/CDF files.
- `data/training_474.csv` and `data/testing_185.csv`: complete training and testing days after minute-level imputation, created by `scripts/02_Imputation_and_Table2.R`.
- `data/No Weather/`, `data/Features/`, and `data/Embedded/`: synthetic wind-vector outputs from the three generator/input settings. Each folder contains `independent.csv` and `consecutive.csv`, with `21 * 23 * 1440` rows of synthetic data for that setting.

The original daily ARM NetCDF/CDF files are not included. The repository starts from the combined CSV above. Users who want to rebuild that CSV from the original ARM order can use `scripts/00_Combining_ARM_data.R` with their own ARM THREDDS `catalog.html` and `thredds-ui.svcs.arm.gov_cookies.txt` files.

## Workflow

Run the scripts from the repository root in numerical order to reproduce the full analysis from the included combined CSV and synthetic outputs.

```r
source("scripts/01_Figure01_Seasonality_and_Zeros.R")
source("scripts/02_Imputation_and_Table2.R")
source("scripts/03_No_Memorization.R")
source("scripts/04_Figures12_and_13_Energy_Scores.R")
```

`00_Combining_ARM_data.R` is optional and only needed if you want to re-download and recombine the original daily ARM files.

### `00_Combining_ARM_data.R`

Uses user-provided `catalog.html` and `thredds-ui.svcs.arm.gov_cookies.txt` files to locate the ARM THREDDS server for the user's data order. It downloads the individual daily files, combines them, and writes:

```text
data/LamontOK_E13_19930721_20250920.csv
```

### `01_Figure01_Seasonality_and_Zeros.R`

Uses the full Lamont E13 CSV to construct Figure 1 and create the June zero-wind CSV used in Appendix A1.

Outputs are saved in:

```text
Seasonality and Zero Analysis/
```

### `02_Imputation_and_Table2.R`

Implements Brownian bridge imputation for short wind-vector gaps, creates Figure A1 and Table 2, summarizes missing-data gaps, and writes the complete-day training and testing sets.

Outputs are saved in:

```text
Imputation Plots/
data/training_474.csv
data/testing_185.csv
```

### `03_No_Memorization.R`

Loads the training data and synthetic data, computes minimum Euclidean distances between observed and synthetic days, and creates the no-memorization analysis for Figures B1 and B2.

Outputs are saved in:

```text
Minimum Euclidean Distance/
```

### `04_Figures12_and_13_Energy_Scores.R`

Loads the training, testing, and synthetic data, computes day-level and hourly energy score residuals, and creates Figures 12, 13, F1, and Table F1.

Outputs are saved in:

```text
Energy Scores/
```

## Weather State Preprocessing

The scripts folder also includes weather-state preprocessing materials:

```text
scripts/README_weather_states
scripts/weather_states_paper.R
```

These files describe and implement preprocessing for discrete weather states used to condition the Time VQ-VAE wind-vector generator. The preprocessing starts from minute-level surface meteorology data, imputes missing barometric pressure and precipitation values with Kalman filtering, aggregates observations into 10-minute blocks, and assigns each block to one of `2^4` factorial weather states.

`weather_states_paper.R` currently expects an imputed minute-level input file at:

```text
data/training_imputed.csv
```

The weather-state label combines four pressure/precipitation indicators:

- pressure-change direction,
- pressure-difference magnitude,
- heavy precipitation,
- any precipitation.

## Requirements

The scripts are written in R. Required packages are installed automatically by the replication scripts if missing:

```r
c(
  "curl", "data.table", "dplyr", "ggplot2", "grid", "gridExtra",
  "lubridate", "MASS", "ncdf4", "purrr", "readr", "rvest",
  "scoringRules", "stringr", "tidyr"
)
```

The weather-state preprocessing script additionally uses:

```r
c("tidyverse", "vroom", "lubridate", "gridExtra", "hms", "imputeTS")
```
