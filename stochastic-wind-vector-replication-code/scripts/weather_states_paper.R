library(tidyverse)
library(vroom)
library(lubridate)
library(gridExtra)
library(hms)
library(imputeTS)



wind_data <- vroom("data/training_imputed.csv", col_names = TRUE) |>
    select(-starts_with("qc_"))

wind_data$atmos_pressure <- na_kalman(wind_data$atmos_pressure)
wind_data$tbrg_precip_total <- na_kalman(wind_data$tbrg_precip_total)


#generate weather states
ws_table <- wind_data |>
    mutate(time10 = floor_date(time, "10 mins")) |>
    mutate(minute10 = minute(time10), day = day(time10), hour = hour(time10), year = year(time10)) |>
    group_by(minute10, hour, day, year) |>
    mutate(atmos_max_hourly = max(atmos_pressure, na.rm = TRUE)) |>
    mutate(atmos_median_hourly = median(atmos_pressure, na.rm = TRUE)) |>
    mutate(atmos_min_hourly = min(atmos_pressure, na.rm = TRUE)) |>
    mutate(atmos_diff = atmos_max_hourly - atmos_min_hourly) |>
    mutate(direction = case_when(abs(atmos_max_hourly - atmos_median_hourly) > 
                                     abs(atmos_median_hourly - atmos_min_hourly) ~ +1, .default = -1)) |>
    mutate(atmos_diff_vector = direction * atmos_diff) |>
    mutate(direction_indicator = case_when(direction == 1 ~ "+", .default = "-")) |>
    mutate(atmosdiff_indicator = case_when(atmos_diff > .02 ~ "+", .default = "-")) |>
    mutate(atmosheavyrain_indicator = case_when(tbrg_precip_total > .25 ~ "+", .default = "-")) |>
    mutate(atmosrain_indicator = case_when(tbrg_precip_total > 0 ~ "+", .default = "-")) |>
    mutate(weather_state = paste0("(", direction_indicator,
                                  ",", atmosdiff_indicator,
                                  ",", atmosheavyrain_indicator,
                                  ",", atmosrain_indicator,
                                  ")"))

table(ws_table$weather_state)

