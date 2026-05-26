# ============================================================
# Wind Data — Imputation
# ============================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads and processes raw wind data
#   2. Builds a complete minute-by-minute timeline for the target date range across all years
#   3. Flags and removes suspect minutes (zero wind speed coinciding with missing temp and rh)
#   4. Summarize data gaps and saves wind_gap_summary.csv
#   5. Imputes gaps up to max_gap minutes using the Brownian bridge approach
#   6. Saves final training and testing full-day CSVs to output_data_dir, named by their day count
#   7. Saves imputation plots, diagnostic CSVs, gap summary,  and a dataset summary table to output_plots_dir
#
# ============================================================
# Edit only this section
# ============================================================

# --- Input data ---
data_file        = "data/LamontOK_E13_19930721_20250920.csv"

# --- Year range for analysis ---
year_start       = 1994
year_end         = 2025

# --- Maximum gap size to impute (minutes) ---
max_gap          = 20

# --- Plot control: how many gaps to produce imputation plots for.
#     Use a positive integer for the N largest gaps, or "all" for every gap
plot_top_n       = 1

# --- Output folders ---
output_data_dir  = "data"              # Final training_N.csv and testing_N.csv go here
output_plots_dir = "Imputation Plots"  # Figures, diagnostics, gap summary, summary table

# ============================================================
# Load/downloaded needed packages
# ============================================================

packages = c("readr", "dplyr", "lubridate", "tidyr",
             "ggplot2", "gridExtra", "purrr", "MASS")

installed = packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])
invisible(lapply(packages, library, character.only = TRUE))

# ============================================================
# Functions
# ============================================================

# --- process_data ---
# Converts wspd/wdir to Easterly/Northerly components and assigns year, minute_of_day,
# set (Training/Testing/Validation/Other), and time_of_day labels to every row.
process_data = function(data) {
  data %>%
    dplyr::mutate(
      time          = as.POSIXct(time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"),
      Easterly      = wspd_vec_mean * sin((wdir_vec_mean * pi) / 180),
      Northerly     = wspd_vec_mean * cos((wdir_vec_mean * pi) / 180),
      year          = year(time),
      minute_of_day = (hour(time) * 60 + minute(time)) + 1,
      set = case_when(
        month(time) == 6 & day(time) <= 21 & year(time) >= 1998 & year(time) <= 2020 ~ "Training",
        month(time) == 6 & day(time) <= 21 & (year(time) <= 1997 | year(time) >= 2021) ~ "Testing",
        month(time) == 6 & day(time) >= 22 & year(time) >= 1998 & year(time) <= 2020  ~ "Validation",
        TRUE ~ "Other"
      ),
      time_of_day = case_when(
        hour(time) >= 0  & hour(time) < 2  ~ "Sunset",
        hour(time) >= 2  & hour(time) < 12 ~ "Night",
        hour(time) >= 12 & hour(time) < 14 ~ "Sunrise",
        hour(time) >= 14 & hour(time) < 24 ~ "Day",
        TRUE ~ NA_character_
      )
    )
}


# --- impute_single_gap ---
# Imputes one gap using a Brownian bridge on Easterly/Northerly components. 
# Requires at least 16 non-NA observations in the 15-minute windows on each side of the gap.
# Returns a named list: success, imputed_wind, before_gap, after_gap, X0_time, Xn_time, sigma_sq.
impute_single_gap = function(gap, complete_wind, random_seed = 21) {
  X0_time      = gap$start - minutes(1)
  Xn_time      = gap$end   + minutes(1)
  impute_times = seq(from = gap$start, to = gap$end, by = "1 min")
  n            = length(impute_times) + 1
  
  before_gap = complete_wind %>%
    dplyr::filter(time >= X0_time - minutes(15), time <= X0_time,
           !is.na(Easterly), !is.na(Northerly))
  
  after_gap = complete_wind %>%
    dplyr::filter(time <= Xn_time + minutes(15), time >= Xn_time,
           !is.na(Easterly), !is.na(Northerly))
  
  if (nrow(before_gap) < 16 || nrow(after_gap) < 16) {
    return(list(success = FALSE, imputed_wind = NULL,
                before_gap = before_gap, after_gap = after_gap,
                X0_time = X0_time, Xn_time = Xn_time, sigma_sq = NA))
  }
  
  calc_diffs = function(df) {
    df %>%
      dplyr::arrange(time) %>%
      dplyr::mutate(
        east_diff  = (lead(Easterly)  - Easterly)^2,
        north_diff = (lead(Northerly) - Northerly)^2,
        total_diff = east_diff + north_diff
      ) %>%
      dplyr::filter(!is.na(total_diff))
  }
  
  sigma_sq = (sum(calc_diffs(before_gap)$total_diff) +
                sum(calc_diffs(after_gap)$total_diff)) / 60
  
  get_boundary = function(t) {
    complete_wind %>%
      dplyr::filter(time == t) %>%
      dplyr::select(Easterly, Northerly) %>%
      as.matrix()
  }
  
  X0 = get_boundary(X0_time)
  Xn = get_boundary(Xn_time)
  d  = Xn - X0
  
  set.seed(random_seed)
  C = matrix(0, nrow = n, ncol = 2)
  for (j in seq_len(n)) {
    C[j, ] = MASS::mvrnorm(1, mu = c(0, 0), Sigma = sigma_sq * diag(2))
  }
  
  C_bar  = colMeans(C)
  Delta  = matrix(0, nrow = n, ncol = 2)
  for (j in seq_len(n)) Delta[j, ] = C[j, ] - C_bar + d / n
  
  X      = matrix(0, nrow = n, ncol = 2)
  X[1, ] = X0 + Delta[1, ]
  for (j in 2:n) X[j, ] = X[j - 1, ] + Delta[j, ]
  
  final_diff = sqrt(sum((X[n, ] - Xn)^2))
  if (final_diff > 0.05)
    warning(sprintf("Final imputed point differs from Xn by %f", final_diff))
  
  imputed_wind = tibble(
    time      = impute_times,
    Easterly  = X[1:(n - 1), 1],
    Northerly = X[1:(n - 1), 2]
  ) %>%
    mutate(
      wspd_vec_mean = sqrt(Easterly^2 + Northerly^2),
      wdir_vec_mean = (atan2(Easterly, Northerly) * 180 / pi) %% 360
    )
  
  list(success = TRUE, imputed_wind = imputed_wind,
       before_gap = before_gap, after_gap = after_gap,
       X0_time = X0_time, Xn_time = Xn_time, sigma_sq = sigma_sq)
}


# --- plot_imputation ---
# Two-panel plot (Northerly / Easterly) showing real data on either side of the gap and the imputed bridge in between
plot_imputation = function(imputed_wind, before_gap, after_gap,
                           X0_time, Xn_time) {
  all_data = bind_rows(
    before_gap   %>% mutate(data_type = "Real"),
    imputed_wind %>% mutate(data_type = "Imputed"),
    after_gap    %>% mutate(data_type = "Real")
  ) %>%
    dplyr::arrange(time) %>%
    dplyr::mutate(is_boundary = time %in% c(X0_time, Xn_time), plot_group = 1L)
  
  plot_title = paste0(
    format(min(imputed_wind$time), "%Y-%m-%d %H:%M"), " to ",
    format(max(imputed_wind$time), "%H:%M")
  )
  
  imp_theme = theme_minimal() +
    theme(
      plot.title       = element_text(size = 22, face = "bold"),
      axis.title       = element_text(size = 18),
      axis.text        = element_text(size = 12),
      legend.position  = "top",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor = element_line(color = "gray95", linewidth = 0.25)
    )
  
  make_panel = function(y_var, real_color, y_label, is_top) {
    ggplot(all_data, aes(x = time, group = plot_group)) +
      geom_line(aes(y = .data[[y_var]]), color = real_color, linewidth = 0.7) +
      geom_point(aes(y = .data[[y_var]], color = data_type), size = 2.5) +
      geom_point(data = filter(all_data, is_boundary),
                 aes(y = .data[[y_var]]),
                 shape = 21, fill = "white", color = "black",
                 size = 3, stroke = 1.2) +
      scale_color_manual(values = c("Real" = real_color, "Imputed" = "black")) +
      labs(
        title    = if (is_top) plot_title else NULL,
        x = "Time", y = y_label, color = "Data Type"
      ) +
      imp_theme
  }
  
  gridExtra::grid.arrange(
    make_panel("Northerly", "orange",    "Northerly (m/s)", is_top = TRUE),
    make_panel("Easterly",  "steelblue", "Easterly (m/s)",  is_top = FALSE),
    nrow = 2, heights = c(1.1, 1)
  )
}


# --- format_year_ranges ---
# Converts a vector of years into a compact range string.
# e.g. c(1994,1995,1996,1997,2021,2022,2023,2024,2025) -> "1994-1997, 2021-2025"
format_year_ranges = function(years) {
  years  = sort(unique(years))
  breaks = c(0, which(diff(years) > 1), length(years))
  ranges = sapply(seq_len(length(breaks) - 1), function(i) {
    seg = years[(breaks[i] + 1):breaks[i + 1]]
    if (length(seg) == 1) as.character(seg)
    else paste0(min(seg), "-", max(seg))
  })
  paste(ranges, collapse = ", ")
}


# --- impute_gaps_controlled ---
# Runs the imputation loop over all gaps <= max_gap_size
# plot_top_n = "all" plots every gap; an integer plots the N largest.
impute_gaps_controlled = function(complete_wind, wind_gap_summary,
                                  max_gap_size = 20,
                                  figures_dir  = "Imputation Plots",
                                  plot_top_n   = "all") {
  
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  
  small_gaps = wind_gap_summary %>%
    dplyr::filter(missing_count <= max_gap_size) %>%
    arrange(start)
  
  if (identical(plot_top_n, "all")) {
    plot_indices = seq_len(nrow(small_gaps))
  } else {
    n_plots      = as.integer(plot_top_n)
    ranked_idx   = small_gaps %>%
      mutate(.idx = row_number()) %>%
      arrange(desc(missing_count), start) %>%
      pull(.idx)
    plot_indices = ranked_idx[seq_len(min(n_plots, length(ranked_idx)))]
  }
  
  imputation_log = tibble(
    gap_id        = integer(),
    start_time    = as.POSIXct(character()),
    end_time      = as.POSIXct(character()),
    missing_count = integer(),
    success       = logical(),
    plot_path     = character(),
    sigma_sq      = numeric()
  )
  
  for (i in seq_len(nrow(small_gaps))) {
    gap = small_gaps[i, ]
    cat(sprintf("  Gap %d / %d  (%d min)  %s\n",
                i, nrow(small_gaps), gap$missing_count,
                as.character(gap$start)))
    
    result = impute_single_gap(gap, complete_wind, random_seed = 21 + i)
    
    if (result$success) {
      plot_filepath = NA_character_
      
      if (i %in% plot_indices) {
        rank_in_plot  = which(plot_indices == i)
        plot_filepath = file.path(
          figures_dir,
          if (rank_in_plot == 1) "figA1.pdf" else paste0("figA1_", rank_in_plot, ".pdf")
        )
        ggsave(plot_filepath,
               plot_imputation(result$imputed_wind, result$before_gap,
                               result$after_gap, result$X0_time, result$Xn_time),
               width = 12, height = 10, dpi = 300)
      }
      
      complete_wind  = complete_wind %>% rows_update(result$imputed_wind, by = "time")
      imputation_log = add_row(imputation_log,
                               gap_id = i, start_time = gap$start, end_time = gap$end,
                               missing_count = gap$missing_count, success = TRUE,
                               plot_path = plot_filepath, sigma_sq = result$sigma_sq)
      cat("    -> imputed\n")
      
    } else {
      imputation_log = add_row(imputation_log,
                               gap_id = i, start_time = gap$start, end_time = gap$end,
                               missing_count = gap$missing_count, success = FALSE,
                               plot_path = NA_character_, sigma_sq = NA_real_)
      cat("    -> skipped (insufficient surrounding data)\n")
    }
  }
  
  # Mark imputed rows
  success_rows  = which(imputation_log$success)
  imputed_times = if (length(success_rows) > 0) {
    as.POSIXct(unique(unlist(lapply(success_rows, function(i)
      as.numeric(seq(imputation_log$start_time[i],
                     imputation_log$end_time[i], by = "1 min"))
    ))), origin = "1970-01-01", tz = "UTC")
  } else {
    as.POSIXct(character(0), tz = "UTC")
  }
  
  complete_wind = complete_wind %>%
    mutate(is_imputed = as.integer(time %in% imputed_times))
  
  attr(complete_wind, "imputation_log") = imputation_log
  
  cat(sprintf("  Done: %d imputed | %d skipped | %d plots saved\n\n",
              sum(imputation_log$success),
              sum(!imputation_log$success),
              sum(!is.na(imputation_log$plot_path))))
  
  return(complete_wind)
}


# ============================================================
# Load and filter data
# ============================================================

wind = readr::read_csv(data_file, show_col_types = FALSE) %>%
  dplyr::mutate(time = as.POSIXct(time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")) %>%
  process_data() %>%
  dplyr::filter(
    (month(time) == 5 & day(time) == 31) |
      (month(time) == 6) |
      (month(time) == 7 & day(time) == 1)
  )

# ============================================================
# Build complete timeline and merge
# ============================================================

years = year_start:year_end

complete_timeline = purrr::map_df(years, function(yr) {
  tibble(time = seq(
    from = ymd_hms(paste0(yr, "-05-31 00:00:00")),
    to   = ymd_hms(paste0(yr, "-07-01 23:59:00")),
    by   = "1 min"
  ))
}) %>%
  mutate(time = as.POSIXct(time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"))

cat("Expected timeline rows:", length(years) * 32 * 1440, "\n")
cat("Actual timeline rows  :", nrow(complete_timeline), "\n\n")

complete_wind = complete_timeline %>%
  left_join(wind, by = "time") %>%
  process_data()

cat("Set sizes (days):\n")
cat("  Training  :", nrow(filter(complete_wind, set == "Training"))   / 1440, "\n")
cat("  Testing   :", nrow(filter(complete_wind, set == "Testing"))    / 1440, "\n")
cat("  Validation:", nrow(filter(complete_wind, set == "Validation")) / 1440, "\n\n")

# Record total days per set now, before imputation, for the summary table
total_training_days = nrow(filter(complete_wind, set == "Training")) / 1440
total_testing_days  = nrow(filter(complete_wind, set == "Testing"))  / 1440

# ============================================================
# Flag and remove suspect minutes
# ============================================================

complete_wind = complete_wind %>%
  dplyr::mutate(
    is_missing = is.na(wspd_vec_mean),
    is_suspect = is.na(temp_mean) & is.na(rh_mean) & !is.na(base_time)
  )

cat("Original missing minutes:", sum(complete_wind$is_missing), "\n")
cat("Suspect minutes          :", sum(complete_wind$is_suspect), "\n\n")

complete_wind = complete_wind %>%
  dplyr::mutate(
    wspd_vec_mean = ifelse(is_suspect, NA_real_, wspd_vec_mean),
    wdir_vec_mean = ifelse(is_suspect, NA_real_, wdir_vec_mean),
    Easterly      = ifelse(is_suspect, NA_real_, Easterly),
    Northerly     = ifelse(is_suspect, NA_real_, Northerly)
  ) %>%
  mutate(is_na = is.na(wspd_vec_mean) & is.na(wdir_vec_mean))

cat("Total NA minutes after suspect removal:", sum(complete_wind$is_na), "\n\n")

# ============================================================
# Gap summary
# ============================================================

wind_gap_summary = complete_wind %>%
  dplyr::filter(is_na) %>%
  dplyr::arrange(time) %>%
  dplyr::mutate(
    time_diff = as.numeric(difftime(time, lag(time), units = "mins")),
    new_gap   = if_else(is.na(time_diff) | time_diff != 1, 1L, 0L),
    gap_group = cumsum(new_gap)
  ) %>%
  group_by(gap_group) %>%
  dplyr::summarise(
    start         = min(time),
    end           = max(time),
    missing_count = n(),
    missing_hours = n() / 60,
    year          = year(min(time)),
    set           = first(set),
    .groups       = "drop"
  ) %>%
  dplyr::mutate(
    start = as.POSIXct(start, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"),
    end   = as.POSIXct(end,   format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  )

dir.create(output_plots_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(wind_gap_summary,
          file.path(output_plots_dir, "wind_gap_summary.csv"),
          row.names = FALSE)
cat("Gap summary saved to:", file.path(output_plots_dir, "wind_gap_summary.csv"), "\n\n")

# ============================================================
# Impute training gaps
# ============================================================

cat("=== Imputing Training set ===\n")

complete_training = complete_wind %>% filter(set == "Training")
training_gaps     = wind_gap_summary %>% filter(set == "Training")

training_wind_imputed = impute_gaps_controlled(
  complete_wind    = complete_training,
  wind_gap_summary = training_gaps,
  max_gap_size     = max_gap,
  figures_dir      = file.path(output_plots_dir),
  plot_top_n       = plot_top_n
)

# ============================================================
# Impute testing gaps
# ============================================================

cat("=== Imputing Testing set ===\n")

complete_testing = complete_wind %>% filter(set == "Testing")
testing_gaps     = wind_gap_summary %>% filter(set == "Testing")

testing_wind_imputed = impute_gaps_controlled(
  complete_wind    = complete_testing,
  wind_gap_summary = testing_gaps,
  max_gap_size     = max_gap,
  #figures_dir      = file.path(output_plots_dir, "Testing"),
  figures_dir      = file.path(output_plots_dir),
  plot_top_n       = 0
)

# ============================================================
# Filter to only full days
# ============================================================

keep_full_days = function(df) {
  df %>%
    mutate(date = as.Date(time)) %>%
    group_by(date) %>%
    filter(n() == 1440,
           !any(is.na(wspd_vec_mean)),
           !any(is.na(wdir_vec_mean))) %>%
    ungroup()
}

training_full = keep_full_days(training_wind_imputed)
testing_full  = keep_full_days(testing_wind_imputed)

n_train = nrow(training_full) / 1440
n_test  = nrow(testing_full)  / 1440

# ============================================================
# Save final csvs
# ============================================================

train_file = file.path(output_data_dir, paste0("training_", n_train, ".csv"))
test_file  = file.path(output_data_dir, paste0("testing_",  n_test,  ".csv"))

write.csv(training_full, train_file, row.names = FALSE)
write.csv(testing_full,  test_file,  row.names = FALSE)

cat("Saved:", train_file, "\n")
cat("Saved:", test_file,  "\n\n")

# ============================================================
# Data summary table (Table 2)
# ============================================================

summary_table = tibble(
  Dataset        = c("Training", "Test"),
  Days           = c("June 1-21", "June 1-21"),
  Years          = c(format_year_ranges(year(training_full$time)),
                     format_year_ranges(year(testing_full$time))),
  `Full Days`    = c(n_train, n_test),
  `Missing Days` = c(total_training_days - n_train,
                     total_testing_days  - n_test)
)

write.csv(summary_table,
          file.path(output_plots_dir, "table2.csv"),
          row.names = FALSE)

cat("Dataset summary:\n")
print(summary_table)
cat("\nSaved to:", file.path(output_plots_dir, "dataset_summary.csv"), "\n")
