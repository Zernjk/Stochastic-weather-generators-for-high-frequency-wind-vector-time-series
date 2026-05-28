# ============================================================
# Energy Score Calculation + Residual Plots
# ============================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads testing, training, and raw synthetic data sets
#   2. Processes synthetic data (add timestamps and match training missing-data pattern)
#   3. Calculates ES(test, training) once, reused for all plots
#   4. Calculates ES(test, synthetic) for all 6 combinations
#   5. Saves combined long-format CSV and produces the combined 3×2 day-level residual plot
#   6. For each combination in individual_plots:
#        a. Individual day-level residual plot
#        b. Calculates hourly energy scores
#        c. Hourly residual plot (even hours, 4×3 grid)
#        d. Saves hourly CSV
#
# ============================================================
# Edit only this section
# ============================================================

# --- Processed data files ---
testing_file  = "data/testing_185.csv"
training_file = "data/training_474.csv"

# --- Raw synthetic data files (Easterly / Northerly columns only) ---
no_weather_indep_file = "data/No Weather/independent.csv"
no_weather_cons_file  = "data/No Weather/consecutive.csv"
features_indep_file   = "data/Features/independent.csv"
features_cons_file    = "data/Features/consecutive.csv"
embedded_indep_file   = "data/Embedded/independent.csv"
embedded_cons_file    = "data/Embedded/consecutive.csv"

# --- Synthetic data processing parameters ---
yearlist    = 1998:2020
days_period = 21
start_month = 6
start_day   = 1

# --- Which combinations get individual day + hourly plots ---
# Add or remove entries as needed
individual_plots = list(
  list(generator = "Embedded", synth_type = "Consecutive")
  # list(generator = "Features", synth_type = "Independent"),
  # list(generator = "No_Weather", synth_type = "Consecutive"),
)

# --- Output ---
output_dir = "Energy Scores"

# --- Plot dimensions ---
day_width        = 8
day_height       = 7
hourly_width     = 16
hourly_height    = 16
combined_width   = 14
combined_height  = 18
dpi              = 300

# --- Text sizes (hourly plot) ---
hourly_title_size   = 28
hourly_axis_title   = 24
hourly_axis_text    = 20
hourly_legend_title = 20
hourly_legend_text  = 18
hourly_legend_sym   = 10
hourly_strip_text   = 24

# --- Text sizes (combined plot) ---
combined_axis_title   = 18
combined_axis_text    = 14
combined_strip_text   = 20
combined_legend_title = 18
combined_legend_text  = 15
combined_legend_sym   = 6

# ============================================================
# Load/download needed packages
# ============================================================

packages = c("readr", "dplyr", "ggplot2", "tidyr", "lubridate",
             "purrr", "scoringRules")

installed = packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])
invisible(lapply(packages, library, character.only = TRUE))

# ============================================================
# Data processing functions
# ============================================================

# Adds timestamps (time, date, hour) and set/day_number columns
# to raw synthetic data, then optionally matches the missing-data pattern from the training dataset.
add_timestamps = function(data, yearlist = 1998:2020,
                          days_period = 21, start_month = 6, start_day = 1) {
  date_fmt = paste0("-",
                    formatC(start_month, width = 2, flag = "0"), "-",
                    formatC(start_day,   width = 2, flag = "0"),
                    " 00:00:00")
  
  make_ts = function(yr) {
    seq(from       = as.POSIXct(paste0(yr, date_fmt), tz = "UTC"),
        by         = "1 min",
        length.out = days_period * 1440)
  }
  
  timestamps = do.call(c, lapply(yearlist, make_ts))
  
  if (nrow(data) > length(timestamps)) {
    extra_years = ceiling((nrow(data) - length(timestamps)) /
                            (days_period * 1440))
    for (k in seq_len(extra_years)) {
      yr = max(yearlist) + k
      timestamps = c(timestamps, make_ts(yr))
    }
  }
  
  data %>%
    dplyr::mutate(
      time   = timestamps[seq_len(dplyr::n())],
      date   = lubridate::date(time),
      hour   = lubridate::hour(time)
    )
}

process_synthetic_data = function(synthetic_data, training_data,
                                  yearlist = 1998:2020, days_period = 21,
                                  start_month = 6, start_day = 1) {
  cat("  Adding timestamps...\n")
  synth = add_timestamps(synthetic_data, yearlist = yearlist,
                         days_period = days_period,
                         start_month = start_month, start_day = start_day)
  
  synth$day_number = ceiling(seq_len(nrow(synth)) / 1440)
  synth$set        = ceiling(synth$day_number /
                               (days_period * length(yearlist)))
  
  expected_rows = days_period * length(yearlist) * 1440
  final_set     = max(synth$set)
  if (sum(synth$set == final_set) != expected_rows) {
    cat("  Removing incomplete final set.\n")
    synth = dplyr::filter(synth, set != final_set)
  }
  
  cat("  Matching training missing-data pattern...\n")
  date_fmt = paste0("-",
                    formatC(start_month, width = 2, flag = "0"), "-",
                    formatC(start_day,   width = 2, flag = "0"),
                    " 00:00:00")
  
  complete_timeline = purrr::map_df(yearlist, function(yr) {
    start = as.POSIXct(paste0(yr, date_fmt), tz = "UTC")
    end   = start + lubridate::days(days_period) - lubridate::seconds(60)
    tibble::tibble(time = seq(start, end, by = "1 min"))
  })
  
  pattern = complete_timeline %>%
    dplyr::left_join(
      dplyr::mutate(training_data, exists = TRUE) %>%
        dplyr::select(time, exists),
      by = "time"
    ) %>%
    dplyr::mutate(is_missing = is.na(exists))
  
  period_minutes = days_period * 1440
  orig_rows = nrow(synth)
  
  synth = synth %>%
    dplyr::group_by(set) %>%
    dplyr::mutate(is_missing = pattern$is_missing) %>%
    dplyr::ungroup() %>%
    dplyr::filter(!is_missing) %>%
    dplyr::select(-is_missing)
  
  cat("  Rows before:", orig_rows, "| after:", nrow(synth),
      "| removed:", orig_rows - nrow(synth), "\n")
  synth
}


# ============================================================
# Energy score functions
# ============================================================

split_by_date = function(df) split(df, df$date)

split_by_hour = function(df) {
  lapply(0:23, function(h) split(df[df$hour == h, ], df$date[df$hour == h]))
}

process_day_data = function(day_data,
                            variables   = c("Easterly", "Northerly"),
                            first_diff  = FALSE,
                            sort_values = FALSE) {
  day_vector = if (first_diff) unlist(lapply(variables, function(v) diff(day_data[[v]])))
  else            unlist(lapply(variables, function(v) day_data[[v]]))
  if (sort_values) day_vector = sort(day_vector)
  day_vector
}

process_hour_data = function(hour_data,
                             variables   = c("Easterly", "Northerly"),
                             first_diff  = FALSE,
                             sort_values = FALSE) {
  hour_vector = if (first_diff) unlist(lapply(variables, function(v) diff(hour_data[[v]])))
  else            unlist(lapply(variables, function(v) hour_data[[v]]))
  if (sort_values) hour_vector = sort(hour_vector)
  hour_vector
}

create_day_df = function(days_list, variables = c("Easterly", "Northerly"),
                         first_diff = FALSE, sort_values = FALSE) {
  row_count = if (first_diff) (1440 - 1) * length(variables)
  else            1440 * length(variables)
  result_df = data.frame(matrix(NA, nrow = row_count, ncol = length(days_list)))
  colnames(result_df) = names(days_list)
  for (i in seq_along(days_list))
    result_df[, i] = process_day_data(days_list[[i]], variables = variables,
                                      first_diff = first_diff, sort_values = sort_values)
  result_df
}

create_hour_df = function(hour_days_list, variables = c("Easterly", "Northerly"),
                          first_diff = FALSE, sort_values = FALSE) {
  row_count = if (first_diff) (60 - 1) * length(variables)
  else            60 * length(variables)
  result_df = data.frame(matrix(NA, nrow = row_count, ncol = length(hour_days_list)))
  colnames(result_df) = names(hour_days_list)
  for (i in seq_along(hour_days_list))
    result_df[, i] = process_hour_data(hour_days_list[[i]], variables = variables,
                                       first_diff = first_diff, sort_values = sort_values)
  result_df
}

# Day-level energy scores: for each test day, ES vs distribution
calculate_energy_scores = function(test_df, dist_df,
                                   variables = c("Easterly", "Northerly")) {
  cat("  Building day matrices...\n")
  test_days = split_by_date(test_df)
  dist_days = split_by_date(dist_df)
  test_mat  = as.matrix(create_day_df(test_days, variables = variables))
  dist_mat  = as.matrix(create_day_df(dist_days, variables = variables))
  
  n_test = ncol(test_mat)
  cat("  Test:", ncol(test_mat), "days | Dist:", ncol(dist_mat), "days\n")
  
  scores = data.frame(
    date         = as.Date(names(test_days)),
    energy_score = numeric(n_test)
  )
  
  for (i in seq_len(n_test)) {
    scores$energy_score[i] = scoringRules::es_sample(
      y = test_mat[, i], dat = dist_mat
    )
    if (i %% 50 == 0) cat("    Processed", i, "of", n_test, "\n")
  }
  cat("  Done.\n")
  scores
}

# Hourly energy scores: for each test day × hour, ES vs distribution
calculate_energy_scores_hourly = function(test_df, dist_df,
                                          variables = c("Easterly", "Northerly")) {
  cat("  Calculating hourly energy scores...\n")
  test_hours = split_by_hour(test_df)
  dist_hours = split_by_hour(dist_df)
  
  all_results = vector("list", 24)
  
  for (h in 0:23) {
    if (h %% 4 == 0) cat("    Hour", h, "...\n")
    test_h = test_hours[[h + 1]]
    dist_h = dist_hours[[h + 1]]
    
    if (length(test_h) == 0 || length(dist_h) == 0) next
    
    test_mat = as.matrix(create_hour_df(test_h, variables = variables))
    dist_mat = as.matrix(create_hour_df(dist_h, variables = variables))
    mode(test_mat) = "numeric";  mode(dist_mat) = "numeric"
    
    n_test = ncol(test_mat)
    dates  = as.Date(names(test_h))
    
    hour_scores = data.frame(
      date         = dates,
      hour         = h,
      energy_score = vapply(seq_len(n_test), function(i)
        scoringRules::es_sample(y = test_mat[, i], dat = dist_mat), numeric(1))
    )
    all_results[[h + 1]] = hour_scores
  }
  
  cat("  Hourly scores done.\n")
  dplyr::bind_rows(all_results)
}


# ============================================================
# Plotting functions
# ============================================================

add_wind_categories = function(df, by_hour = FALSE) {
  make_cats = function(d) {
    wq = quantile(d$avg_wind_speed, probs = c(0, 0.1, 0.5, 0.9, 1), na.rm = TRUE)
    d %>% dplyr::mutate(
      wind_color_cat = cut(avg_wind_speed, breaks = wq,
                           labels = c("Q1", "Q2", "Q3", "Q4"),
                           include.lowest = TRUE),
      wind_shape_cat = cut(avg_wind_speed,
                           breaks = quantile(avg_wind_speed,
                                             probs = c(0, 0.5, 1), na.rm = TRUE),
                           labels = c("Low", "High"), include.lowest = TRUE)
    )
  }
  if (by_hour) {
    df %>% dplyr::group_by(hour) %>%
      dplyr::group_modify(~ make_cats(.x)) %>% dplyr::ungroup()
  } else {
    make_cats(df)
  }
}

wind_color_scale = function() {
  scale_color_manual(
    name   = "Average Wind\nSpeed",
    values = c("Q1" = "#d31f11", "Q2" = "darkgray",
               "Q3" = "#f47a00", "Q4" = "#007191"),
    labels = c("Low", "Low-Mid", "Mid-High", "High")
  )
}

wind_shape_scale = function() {
  scale_shape_manual(
    name   = "Wind Speed\nRange",
    values = c("Low" = 4, "High" = 3),
    labels = c("Below Median", "Above Median")
  )
}

color_strip_text = function(g) {
  strip_idx = which(grepl("strip", g$layout$name))
  for (i in strip_idx) {
    label = tryCatch(
      g$grobs[[i]]$grobs[[1]]$children[[2]]$children[[1]]$label,
      error = function(e) NA_character_
    )
    h = suppressWarnings(as.numeric(gsub("Hour ", "", label)))
    if (!is.na(h)) {
      col = if (h %in% 2:11) "#00008B" else if (h %in% 14:23) "#006400" else "black"
      g$grobs[[i]]$grobs[[1]]$children[[2]]$children[[1]]$gp$col = col
    }
  }
  g
}

make_day_residual_plot = function(plot_data, display_name) {
  x_max = ceiling(max(plot_data$es_train, na.rm = TRUE) / 50) * 50
  y_min = floor(min(plot_data$es_diff,   na.rm = TRUE)) - 5
  y_max = ceiling(max(plot_data$es_diff, na.rm = TRUE)) + 5
  
  ggplot(plot_data, aes(x = es_train, y = es_diff,
                        color = wind_color_cat, shape = wind_shape_cat)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black",  linewidth = 1.2) +
    geom_vline(xintercept = 0, linetype = "solid",  color = "gray30", linewidth = 0.6) +
    geom_point(size = 4, alpha = 0.7, stroke = 1.5) +
    wind_color_scale() + wind_shape_scale() +
    scale_x_continuous(limits = c(0, x_max)) +
    scale_y_continuous(limits = c(y_min, y_max)) +
    labs(title = display_name,
         x = "Energy Score with Training",
         y = "Difference (Synthetic - Training)") +
    guides(color = guide_legend(override.aes = list(size = 5, stroke = 2)),
           shape = guide_legend(override.aes = list(size = 5, stroke = 2))) +
    theme_minimal() +
    theme(
      plot.title        = element_text(hjust = 0.5, size = 20, face = "bold"),
      axis.title        = element_text(size = 18, face = "bold"),
      axis.text         = element_text(size = 16),
      legend.position   = "right",
      legend.title      = element_text(size = 16, face = "bold"),
      legend.text       = element_text(size = 14),
      legend.key.height = unit(1, "cm"),
      legend.key.width  = unit(0.8, "cm"),
      panel.grid.minor  = element_blank()
    )
}

make_hourly_residual_plot = function(plot_data, display_name,
                                     title_size, axis_title, axis_text,
                                     legend_title, legend_text, legend_sym,
                                     strip_text) {
  x_max = ceiling(max(plot_data$es_train, na.rm = TRUE))
  y_min = min(plot_data$es_diff, na.rm = TRUE) - 1
  y_max = max(plot_data$es_diff, na.rm = TRUE) + 1
  
  p = ggplot(plot_data, aes(x = es_train, y = es_diff,
                            color = wind_color_cat, shape = wind_shape_cat)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black",  linewidth = 1) +
    geom_vline(xintercept = 0, linetype = "solid",  color = "gray50", linewidth = 0.5) +
    geom_point(size = 5, alpha = 0.7, stroke = 2) +
    facet_wrap(~ hour_label, ncol = 3, nrow = 4) +
    wind_color_scale() + wind_shape_scale() +
    scale_x_continuous(limits = c(0, x_max), breaks = seq(0, 100, by = 25)) +
    scale_y_continuous(limits = c(y_min, y_max)) +
    labs(title = display_name,
         x = "Energy Score with Training",
         y = "Difference (Synthetic - Training)") +
    guides(color = guide_legend(override.aes = list(size = legend_sym, stroke = 2)),
           shape = guide_legend(override.aes = list(size = legend_sym, stroke = 2))) +
    theme_minimal() +
    theme(
      plot.title        = element_text(hjust = 0.5, size = title_size, face = "bold"),
      axis.title        = element_text(size = axis_title,  face = "bold"),
      axis.text         = element_text(size = axis_text),
      legend.position   = "top",
      legend.direction  = "horizontal",
      legend.title      = element_text(size = legend_title, face = "bold"),
      legend.text       = element_text(size = legend_text),
      legend.key.width  = unit(1.2, "cm"),
      legend.key.height = unit(0.8, "cm"),
      strip.text        = element_text(size = strip_text,  face = "bold"),
      panel.spacing     = unit(0.8, "lines"),
      plot.margin       = margin(10, 10, 10, 10)
    )
  
  color_strip_text(ggplotGrob(p))
}

make_combined_day_plot = function(all_day_data,
                                  axis_title, axis_text, strip_text,
                                  legend_title, legend_text, legend_sym) {
  plot_data = all_day_data %>%
    dplyr::group_by(generator, synth_type) %>%
    dplyr::group_modify(function(d, k) {
      wq = quantile(d$avg_wind_speed, probs = c(0, 0.1, 0.5, 0.9, 1), na.rm = TRUE)
      d %>% dplyr::mutate(
        wind_color_cat = cut(avg_wind_speed, breaks = wq,
                             labels = c("Q1", "Q2", "Q3", "Q4"),
                             include.lowest = TRUE),
        wind_shape_cat = cut(avg_wind_speed,
                             breaks = quantile(avg_wind_speed,
                                               probs = c(0, 0.5, 1), na.rm = TRUE),
                             labels = c("Low", "High"), include.lowest = TRUE)
      )
    }) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      generator  = factor(generator,
                          levels = c("Embedded", "Features", "No_Weather"),
                          labels = c("Embedded", "Features", "No Weather")),
      synth_type = factor(synth_type, levels = c("Independent", "Consecutive"))
    ) %>%
    dplyr::arrange(generator, synth_type, date)
  
  y_min = floor(min(plot_data$es_diff,   na.rm = TRUE)) - 5
  y_max = ceiling(max(plot_data$es_diff, na.rm = TRUE)) + 5
  x_max = ceiling(max(plot_data$es_train, na.rm = TRUE) / 50) * 50
  
  pct_summary = plot_data %>%
    dplyr::mutate(pct_diff = 100 * es_diff / es_train) %>%
    dplyr::group_by(generator, synth_type) %>%
    dplyr::summarise(
      n      = dplyr::n(),
      min    = round(min(pct_diff,            na.rm = TRUE), 2),
      max    = round(max(pct_diff,            na.rm = TRUE), 2),
      mean   = round(mean(pct_diff,           na.rm = TRUE), 2),
      median = round(median(pct_diff,         na.rm = TRUE), 2),
      Q1     = round(quantile(pct_diff, 0.25, na.rm = TRUE), 2),
      Q3     = round(quantile(pct_diff, 0.75, na.rm = TRUE), 2),
      IQR    = round(IQR(pct_diff,            na.rm = TRUE), 2),
      .groups = "drop"
    )
  
  cat("\nPercentage of training energy score summary:\n")
  print(pct_summary, n = Inf)
  
  p = ggplot(plot_data,
             aes(x = es_train, y = es_diff,
                 color = wind_color_cat, shape = wind_shape_cat)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black",  linewidth = 1.2) +
    geom_vline(xintercept = 0, linetype = "solid",  color = "gray30", linewidth = 0.6) +
    geom_point(size = 5, alpha = 0.7, stroke = 1.5) +
    facet_grid(rows = vars(generator), cols = vars(synth_type)) +
    wind_color_scale() + wind_shape_scale() +
    scale_x_continuous(limits = c(0, x_max)) +
    scale_y_continuous(limits = c(y_min, y_max)) +
    labs(x = "Energy Score with Training",
         y = "Difference (Synthetic - Training)") +
    guides(color = guide_legend(override.aes = list(size = legend_sym, stroke = 2)),
           shape = guide_legend(override.aes = list(size = legend_sym, stroke = 2))) +
    theme_minimal() +
    theme(
      axis.title        = element_text(size = axis_title, face = "bold"),
      axis.text         = element_text(size = axis_text),
      strip.text.x      = element_text(size = strip_text, face = "bold"),
      strip.text.y      = element_text(size = strip_text, face = "bold"),
      strip.background  = element_rect(fill = "gray92", color = NA),
      legend.position   = "right",
      legend.title      = element_text(size = legend_title, face = "bold"),
      legend.text       = element_text(size = legend_text),
      legend.key.height = unit(1.4, "cm"),
      legend.key.width  = unit(1.1, "cm"),
      panel.grid.minor  = element_blank(),
      panel.spacing     = unit(1.0, "lines"),
      plot.margin       = margin(12, 12, 12, 12)
    )
  
  list(plot = p, pct_summary = pct_summary,
       plot_data = dplyr::select(all_day_data,
                                 date, es_train, es_synth,
                                 avg_wind_speed, es_diff,
                                 generator, synth_type))
}

# ============================================================
# Output setup
# ============================================================

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Load and process data
# ============================================================

cat("Loading testing and training data...\n")
testing  = readr::read_csv(testing_file,  show_col_types = FALSE) %>%
  dplyr::mutate(time = as.POSIXct(time, tz = "UTC"),
                date = as.Date(date),
                hour = lubridate::hour(time))
training = readr::read_csv(training_file, show_col_types = FALSE) %>%
  dplyr::mutate(time = as.POSIXct(time, tz = "UTC"),
                date = as.Date(date),
                hour = lubridate::hour(time))

# Average wind speed per test day (for plot coloring)
avg_wind = testing %>%
  dplyr::group_by(date) %>%
  dplyr::summarise(
    avg_wind_speed = if ("wspd_vec_mean" %in% names(.)) {
      mean(wspd_vec_mean, na.rm = TRUE)
    } else {
      mean(sqrt(Easterly^2 + Northerly^2), na.rm = TRUE)
    },
    .groups = "drop"
  )

# Load and process synthetic data
synth_spec = list(
  list(file = no_weather_indep_file, generator = "No_Weather", synth_type = "Independent"),
  list(file = no_weather_cons_file,  generator = "No_Weather", synth_type = "Consecutive"),
  list(file = features_indep_file,   generator = "Features",   synth_type = "Independent"),
  list(file = features_cons_file,    generator = "Features",   synth_type = "Consecutive"),
  list(file = embedded_indep_file,   generator = "Embedded",   synth_type = "Independent"),
  list(file = embedded_cons_file,    generator = "Embedded",   synth_type = "Consecutive")
)
synth_spec = Filter(function(x) !is.null(x$file), synth_spec)

synth_data = lapply(synth_spec, function(s) {
  cat("\nLoading and processing:", s$generator, s$synth_type, "\n")
  raw = readr::read_csv(s$file, col_names = FALSE,
                        show_col_types = FALSE) %>%
    dplyr::select(1:2) %>%
    dplyr::rename(Easterly = 1, Northerly = 2)
  
  process_synthetic_data(
    synthetic_data = raw,
    training_data  = training,
    yearlist       = yearlist,
    days_period    = days_period,
    start_month    = start_month,
    start_day      = start_day
  )
})

# ============================================================
# Calculate energy scores
# ============================================================

cat("\n=== Calculating ES(test, training) ===\n")
es_train_df = calculate_energy_scores(testing, training)

all_day_data = list()

for (i in seq_along(synth_spec)) {
  s = synth_spec[[i]]
  cat("\n=== ES(test, synthetic):", s$generator, s$synth_type, "===\n")
  
  es_synth_df = calculate_energy_scores(testing, synth_data[[i]])
  
  day_data = es_train_df %>%
    dplyr::rename(es_train = energy_score) %>%
    dplyr::left_join(
      dplyr::rename(es_synth_df, es_synth = energy_score),
      by = "date"
    ) %>%
    dplyr::left_join(avg_wind, by = "date") %>%
    dplyr::mutate(
      es_diff    = es_synth - es_train,
      generator  = s$generator,
      synth_type = s$synth_type
    ) %>%
    dplyr::arrange(date)
  
  all_day_data[[paste(s$generator, s$synth_type)]] = day_data
}

# ============================================================
# Combined 3 × 2 Day-level plot (Figure F1)
# ============================================================

cat("\n=== Creating combined 3×2 day-level plot ===\n")

combined_long = dplyr::bind_rows(all_day_data)

result = make_combined_day_plot(
  all_day_data  = combined_long,
  axis_title    = combined_axis_title,
  axis_text     = combined_axis_text,
  strip_text    = combined_strip_text,
  legend_title  = combined_legend_title,
  legend_text   = combined_legend_text,
  legend_sym    = combined_legend_sym
)

readr::write_csv(result$plot_data,
                 file.path(output_dir, "AppendixF_es_day_residuals_combined.csv"))
readr::write_csv(result$pct_summary,
                 file.path(output_dir, "tableF1.csv"))

ggsave(file.path(output_dir, "figF1.pdf"),
       result$plot, width = combined_width, height = combined_height, dpi = dpi)

cat("Combined plot saved to:", file.path(output_dir, "figF1.pdf"), "\n")

# ============================================================
# Individual day + hourly plots for requested combinations
# ============================================================

for (cfg in individual_plots) {
  key = paste(cfg$generator, cfg$synth_type)
  
  if (!key %in% names(all_day_data)) {
    warning("No data found for ", key, " — check individual_plots config")
    next
  }
  
  cat("\n=== Individual plots:", cfg$generator, cfg$synth_type, "===\n")
  
  gen_dir    = file.path(output_dir, cfg$generator)
  gen_lower  = tolower(cfg$generator)
  type_lower = tolower(cfg$synth_type)
  display    = paste0(gsub("_", " ", cfg$generator), " - ", cfg$synth_type)
  
  # Day-level individual plot
  day_data = all_day_data[[key]] %>%
    add_wind_categories(by_hour = FALSE)
  
  day_pdf = if (cfg$generator == "Embedded" && cfg$synth_type == "Consecutive") {
    file.path(output_dir, "fig12.pdf")
  } else {
    file.path(gen_dir, paste0("figure12_", gen_lower, "_es_day_residuals_", type_lower, ".pdf"))
  }
  ggsave(day_pdf, make_day_residual_plot(day_data, display),
         width = day_width, height = day_height, dpi = dpi)
  cat("  Day plot saved to:", day_pdf, "\n")
  
  # Find the synthetic data for this combination
  synth_idx = which(sapply(synth_spec, function(s)
    s$generator == cfg$generator && s$synth_type == cfg$synth_type))
  
  if (length(synth_idx) == 0) {
    warning("  Synthetic data not found for hourly calculation — skipping hourly")
    next
  }
  
  # Hourly energy scores
  cat("  Calculating hourly energy scores...\n")
  es_synth_hourly = calculate_energy_scores_hourly(testing, synth_data[[synth_idx]])
  es_train_hourly = calculate_energy_scores_hourly(testing, training)
  
  avg_wind_hourly = testing %>%
    dplyr::group_by(date, hour) %>%
    dplyr::summarise(
      avg_wind_speed = if ("wspd_vec_mean" %in% names(.)) {
        mean(wspd_vec_mean, na.rm = TRUE)
      } else {
        mean(sqrt(Easterly^2 + Northerly^2), na.rm = TRUE)
      },
      .groups = "drop"
    )
  
  hourly_data = es_train_hourly %>%
    dplyr::rename(es_train = energy_score) %>%
    dplyr::left_join(
      dplyr::rename(es_synth_hourly, es_synth = energy_score),
      by = c("date", "hour")
    ) %>%
    dplyr::left_join(avg_wind_hourly, by = c("date", "hour")) %>%
    dplyr::mutate(es_diff = es_synth - es_train) %>%
    dplyr::filter(hour %% 2 == 0) %>%
    dplyr::arrange(date, hour) %>%
    add_wind_categories(by_hour = TRUE) %>%
    dplyr::mutate(
      hour_label = factor(paste0("Hour ", hour),
                          levels = paste0("Hour ", seq(0, 22, by = 2)))
    )
  
  # Save hourly CSV (clean columns)
  hour_csv = if (cfg$generator == "Embedded" && cfg$synth_type == "Consecutive") {
    file.path(output_dir, "fig13.csv")
  } else {
    file.path(gen_dir, paste0("figure13_", gen_lower, "_es_hourly_residuals_", type_lower, ".csv"))
  }
  
  readr::write_csv(
    dplyr::select(hourly_data, date, hour, es_train, es_synth,
                  avg_wind_speed, es_diff),
    hour_csv
  )
  cat("  Hourly CSV saved to:", hour_csv, "\n")
  
  # Hourly plot
  hour_grob = make_hourly_residual_plot(
    plot_data    = hourly_data,
    display_name = display,
    title_size   = hourly_title_size,
    axis_title   = hourly_axis_title,
    axis_text    = hourly_axis_text,
    legend_title = hourly_legend_title,
    legend_text  = hourly_legend_text,
    legend_sym   = hourly_legend_sym,
    strip_text   = hourly_strip_text
  )
  
  hour_pdf = if (cfg$generator == "Embedded" && cfg$synth_type == "Consecutive") {
    file.path(output_dir, "fig13.pdf")
  } else {
    file.path(gen_dir, paste0("figure13_", gen_lower, "_es_hourly_residuals_", type_lower, ".pdf"))
  }
  ggsave(hour_pdf, hour_grob,
         width = hourly_width, height = hourly_height, dpi = dpi)
  cat("  Hourly plot saved to:", hour_pdf, "\n")
}

cat("\nDone.\n")
