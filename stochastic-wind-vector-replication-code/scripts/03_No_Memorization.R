# ============================================================
# Minimum Euclidean Distance
# ============================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads training and all synthetic data sets
#   2. Computes minimum Euclidean distances (Train vs Train, Train vs each synthetic combination)
#   3. Assembles one long-format CSV and produces the combined boxplot
#   4. Saves the N globally closest training/synthetic pairs
#
# ============================================================
# Edit only this section
# ============================================================

# --- Training data ---
training_file = "data/training_474.csv"

# --- Synthetic data files (set any path to NULL to skip that combination) ---
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

# --- Wind vector comparison plots ---
# N globally closest training/synthetic pairs across all combinations
n_wind_vector_plots = 1

# --- Output: all outputs go in one folder ---
# Boxplot PDF, long CSV, and wind vector PDFs are all saved here
output_dir      = "Minimum Euclidean Distance"
output_filename = "figB1"
plot_width      = 16
plot_height     = 9
dpi             = 300

# ============================================================
# Load/download needed packages
# ============================================================

packages = c("readr", "dplyr", "ggplot2", "tidyr", "lubridate", "purrr")

installed = packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])
invisible(lapply(packages, library, character.only = TRUE))

# ============================================================
# Functions
# ============================================================

split_by_date = function(df) split(df, df$date)

process_day_data = function(day_data,
                            variables   = c("Easterly", "Northerly"),
                            first_diff  = FALSE,
                            sort_values = FALSE) {
  if (!all(variables %in% names(day_data)))
    stop("Missing variables: ", paste(setdiff(variables, names(day_data)), collapse = ", "))
  day_vector = if (first_diff) unlist(lapply(variables, function(v) diff(day_data[[v]])))
  else            unlist(lapply(variables, function(v) day_data[[v]]))
  if (sort_values) day_vector = sort(day_vector)
  day_vector
}

create_day_df = function(days_list,
                         variables   = c("Easterly", "Northerly"),
                         first_diff  = FALSE,
                         sort_values = FALSE) {
  num_vars  = length(variables)
  row_count = if (first_diff) (1440 - 1) * num_vars else 1440 * num_vars
  result_df = data.frame(matrix(NA, nrow = row_count, ncol = length(days_list)))
  colnames(result_df) = names(days_list)
  for (i in seq_along(days_list))
    result_df[, i] = process_day_data(days_list[[i]], variables = variables,
                                      first_diff = first_diff, sort_values = sort_values)
  result_df
}

euclidean_distance = function(vec1, vec2) sqrt(sum((vec1 - vec2)^2))

calculate_min_distances = function(set1, set2,
                                   variables    = c("Easterly", "Northerly"),
                                   first_diff   = FALSE,
                                   sort_values  = FALSE,
                                   exclude_self = FALSE) {
  days1 = split_by_date(set1);  days2 = split_by_date(set2)
  mat1  = create_day_df(days1, variables = variables,
                        first_diff = first_diff, sort_values = sort_values)
  mat2  = create_day_df(days2, variables = variables,
                        first_diff = first_diff, sort_values = sort_values)
  dates1 = names(days1);  dates2 = names(days2)
  min_distances = numeric(ncol(mat1));  closest_dates = character(ncol(mat1))
  for (i in seq_len(ncol(mat1))) {
    dists = vapply(seq_len(ncol(mat2)),
                   function(j) euclidean_distance(mat1[, i], mat2[, j]), numeric(1))
    if (exclude_self && dates1[i] %in% dates2)
      dists[which(dates2 == dates1[i])] = Inf
    best             = which.min(dists)
    min_distances[i] = dists[best]
    closest_dates[i] = dates2[best]
  }
  data.frame(date = dates1, min_distance = min_distances,
             closest_date = closest_dates, stringsAsFactors = FALSE)
}


# --- add_timestamps ---
# Assigns minute-by-minute timestamps (time, date, hour) to raw synthetic data based on the training year/period structure
add_timestamps = function(data, yearlist = 1998:2020,
                          days_period = 21, start_month = 6, start_day = 1) {
  date_fmt = paste0("-",
                    formatC(start_month, width = 2, flag = "0"), "-",
                    formatC(start_day,   width = 2, flag = "0"),
                    " 00:00:00")
  make_ts = function(yr) {
    seq(from = as.POSIXct(paste0(yr, date_fmt), tz = "UTC"),
        by = "1 min", length.out = days_period * 1440)
  }
  timestamps = do.call(c, lapply(yearlist, make_ts))
  if (nrow(data) > length(timestamps)) {
    extra = ceiling((nrow(data) - length(timestamps)) / (days_period * 1440))
    for (k in seq_len(extra))
      timestamps = c(timestamps, make_ts(max(yearlist) + k))
  }
  data %>%
    dplyr::mutate(
      time = timestamps[seq_len(dplyr::n())],
      date = lubridate::date(time),
      hour = lubridate::hour(time)
    )
}

# --- process_synthetic_data ---
# Adds timestamps and then removes rows that correspond to missing minutes in the training data.
process_synthetic_data = function(synthetic_data, training_data,
                                  yearlist = 1998:2020, days_period = 21,
                                  start_month = 6, start_day = 1) {
  synth = add_timestamps(synthetic_data, yearlist = yearlist,
                         days_period = days_period,
                         start_month = start_month, start_day = start_day)
  
  synth$day_number = ceiling(seq_len(nrow(synth)) / 1440)
  synth$set        = ceiling(synth$day_number / (days_period * length(yearlist)))
  
  expected_rows = days_period * length(yearlist) * 1440
  final_set     = max(synth$set)
  if (sum(synth$set == final_set) != expected_rows) {
    synth = dplyr::filter(synth, set != final_set)
  }
  
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
      dplyr::mutate(training_data, exists = TRUE) %>% dplyr::select(time, exists),
      by = "time"
    ) %>%
    dplyr::mutate(is_missing = is.na(exists))
  
  synth %>%
    dplyr::group_by(set) %>%
    dplyr::mutate(is_missing = pattern$is_missing) %>%
    dplyr::ungroup() %>%
    dplyr::filter(!is_missing) %>%
    dplyr::select(-is_missing)
}


# --- create_day_comparison_plot ---
# Two-panel time series comparing one training day (top) to the
# closest synthetic day (bottom). Background shading:
#   gray   = night   (2 am – noon)
#   yellow = daytime (2 pm – midnight)
create_day_comparison_plot = function(day1_data, day2_data,
                                      date1, date2,
                                      distance,
                                      generator  = "",
                                      synth_type = "") {
  if (nrow(day1_data) != 1440)
    warning("Training day has ", nrow(day1_data), " rows (expected 1440)")
  if (nrow(day2_data) != 1440)
    warning("Synthetic day has ", nrow(day2_data), " rows (expected 1440)")
  
  gen_clean    = gsub("_", " ", generator)
  panel1_label = paste0("Training Data for ",
                        format(as.Date(date1), "%B %d, %Y"))
  panel2_label = paste0(gen_clean, " - ", synth_type)
  
  prep = function(df, label) {
    df %>%
      dplyr::select(Easterly, Northerly) %>%
      dplyr::mutate(minute = seq_len(dplyr::n())) %>%
      tidyr::pivot_longer(cols = c("Easterly", "Northerly"),
                          names_to  = "component",
                          values_to = "wind_speed") %>%
      dplyr::mutate(panel = label)
  }
  
  plot_data = dplyr::bind_rows(prep(day1_data, panel1_label),
                               prep(day2_data, panel2_label)) %>%
    dplyr::mutate(panel = factor(panel, levels = c(panel1_label, panel2_label)))
  
  shading = data.frame(
    xmin       = c(2 * 60, 14 * 60),
    xmax       = c(12 * 60, 24 * 60),
    fill_color = c("gray80", "#FFE699"),
    ymin = -Inf, ymax = Inf
  )
  
  ggplot(plot_data, aes(x = minute, y = wind_speed, color = component)) +
    geom_rect(data = shading,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                  fill = fill_color),
              alpha = 0.5, inherit.aes = FALSE) +
    scale_fill_identity() +
    geom_line(linewidth = 1.0) +
    facet_wrap(~ panel, ncol = 1, scales = "free_y") +
    scale_color_manual(values = c("Easterly"  = "steelblue",
                                  "Northerly" = "orange"),
                       name = NULL) +
    scale_x_continuous(breaks = seq(0, 1440, by = 240),
                       labels = seq(0, 24,   by = 4),
                       limits = c(0, 1450), expand = c(0, 0)) +
    labs(title = paste0("Euclidean distance between days: ",
                        round(distance, 2)),
         x = "Hour", y = "Wind Speed (m/s)") +
    guides(color = guide_legend(override.aes = list(linewidth = 3))) +
    theme_minimal() +
    theme(
      plot.title        = element_text(hjust = 0.5, size = 13, face = "bold"),
      axis.title        = element_text(size = 12, face = "bold"),
      axis.text         = element_text(size = 11),
      legend.position   = "bottom",
      legend.text       = element_text(size = 14, face = "bold"),
      legend.key.size   = unit(1.5, "cm"),
      legend.box.margin = margin(t = 10),
      strip.text        = element_text(size = 11, face = "bold"),
      strip.background  = element_blank(),
      panel.grid.minor  = element_blank()
    )
}

# ============================================================
# Setup output folder
# ============================================================

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Load data
# ============================================================

cat("Loading data...\n")
training = readr::read_csv(training_file, show_col_types = FALSE) %>%
  dplyr::mutate(
    time = as.POSIXct(time, tz = "UTC"),
    date = as.Date(date)
  )

synth_spec = list(
  list(file = no_weather_indep_file, generator = "No Weather", synth_type = "Independent"),
  list(file = no_weather_cons_file,  generator = "No Weather", synth_type = "Consecutive"),
  list(file = features_indep_file,   generator = "Features",   synth_type = "Independent"),
  list(file = features_cons_file,    generator = "Features",   synth_type = "Consecutive"),
  list(file = embedded_indep_file,   generator = "Embedded",   synth_type = "Independent"),
  list(file = embedded_cons_file,    generator = "Embedded",   synth_type = "Consecutive")
)
synth_spec = Filter(function(x) !is.null(x$file), synth_spec)

synth_data = lapply(synth_spec, function(s) {
  cat("  Loading and processing:", s$generator, s$synth_type, "\n")
  raw = readr::read_csv(s$file, col_names = FALSE, show_col_types = FALSE) %>%
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
# Compute minimum distances
# ============================================================

cat("\nComputing minimum distances...\n")

cat("  Train vs Train (self excluded)...\n")
train_train = calculate_min_distances(training, training, exclude_self = TRUE) %>%
  dplyr::mutate(comparison = "Train vs Train",
                generator  = "Baseline", synth_type = "Baseline")

synth_distances = lapply(seq_along(synth_spec), function(i) {
  s = synth_spec[[i]]
  cat(" ", s$generator, s$synth_type, "...\n")
  calculate_min_distances(training, synth_data[[i]]) %>%
    dplyr::mutate(comparison = "Train vs Synth",
                  generator  = s$generator, synth_type = s$synth_type)
})

# ============================================================
# Build and save long csv
# ============================================================

long_data = dplyr::bind_rows(train_train, dplyr::bind_rows(synth_distances)) %>%
  dplyr::select(date, min_distance, closest_date, comparison, generator, synth_type)

csv_path = file.path(output_dir, paste0(output_filename, ".csv"))
readr::write_csv(long_data, csv_path)
cat("\nLong-format CSV saved to:", csv_path, "\n")

# ============================================================
# Wind vector comparison plots
# (globally closest N pairs across all combinations)
# ============================================================

cat("\nGenerating wind vector comparison plots (global top", n_wind_vector_plots, ")...\n")

# Tag each synth_distances entry with its list index so we can look up the right synth_data[[i]] after sorting globally
all_synth_dist = dplyr::bind_rows(
  lapply(seq_along(synth_spec), function(i) {
    synth_distances[[i]] %>%
      dplyr::mutate(
        generator  = synth_spec[[i]]$generator,
        synth_type = synth_spec[[i]]$synth_type,
        spec_idx   = i
      )
  })
)

top_pairs = all_synth_dist %>%
  dplyr::arrange(min_distance) %>%
  dplyr::slice_head(n = n_wind_vector_plots)

for (j in seq_len(nrow(top_pairs))) {
  pair = top_pairs[j, ]
  i    = pair$spec_idx
  
  train_day = training %>% dplyr::filter(date == as.Date(pair$date))
  synth_day = synth_data[[i]] %>% dplyr::filter(date == as.Date(pair$closest_date))
  
  if (nrow(train_day) == 0 || nrow(synth_day) == 0) {
    warning("  Pair ", j, ": day data not found — skipping")
    next
  }
  
  p = create_day_comparison_plot(
    day1_data  = train_day,
    day2_data  = synth_day,
    date1      = pair$date,
    date2      = pair$closest_date,
    distance   = pair$min_distance,
    generator  = pair$generator,
    synth_type = pair$synth_type
  )
  
  plot_file = file.path(output_dir, if (j == 1) "figB2.pdf" else paste0("figB2_", j, ".pdf"))
  ggsave(plot_file, p, width = 10, height = 8, dpi = dpi)
  cat("  Saved:", plot_file, "\n")
}

# ============================================================
# Build plot data
# ============================================================

generator_order = c("No Weather", "Features", "Embedded")

baseline_plot = long_data %>%
  dplyr::filter(generator == "Baseline") %>%
  dplyr::mutate(x_pos = ifelse(comparison == "Train vs Train", 1, 2))

synth_plot = long_data %>%
  dplyr::filter(generator != "Baseline") %>%
  dplyr::mutate(
    generator = factor(generator, levels = generator_order),
    g_idx     = as.integer(generator),
    x_pos     = ifelse(synth_type == "Independent", 2 + g_idx, 5 + g_idx)
  ) %>%
  dplyr::select(-g_idx)

plot_data = dplyr::bind_rows(baseline_plot, synth_plot) %>%
  dplyr::mutate(
    x_pos     = factor(x_pos, levels = 1:8),
    generator = factor(generator, levels = c("Baseline", generator_order))
  )

x_labels = c(
  "Train vs\nTrain",        # 1
  "",                       # 2
  "Train vs\nIndependent",  # 3
  "",                       # 4
  "",                       # 5
  "Train vs\nConsecutive",  # 6
  ""                        # 7
)

generator_colors = c(
  "Baseline"   = "gray50",
  "No Weather" = "red",
  "Features"   = "steelblue",
  "Embedded"   = "orange"
)

# ============================================================
# Boxplot
# ============================================================

cat("Creating combined boxplot...\n")

p = ggplot(plot_data, aes(x = x_pos, y = min_distance, fill = generator)) +
  geom_boxplot(outlier.size = 3, outlier.alpha = 0.6) +
  scale_fill_manual(values = generator_colors, name = "Generator") +
  scale_x_discrete(labels = x_labels) +
  labs(x = NULL, y = "Minimum Euclidean Distance") +
  theme_minimal() +
  theme(
    axis.title.y       = element_text(size = 18, face = "bold"),
    axis.text.y        = element_text(size = 16),
    axis.text.x        = element_text(size = 16, face = "bold", lineheight = 0.9),
    legend.position    = "bottom",
    legend.title       = element_text(size = 20, face = "bold"),
    legend.text        = element_text(size = 18),
    legend.key.size    = unit(2,   "cm"),
    legend.key.height  = unit(1.2, "cm"),
    legend.key.width   = unit(1.5, "cm"),
    legend.spacing.x   = unit(0.5, "cm"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(10, 10, 10, 10)
  )

plot_path = file.path(output_dir, paste0(output_filename, ".pdf"))
ggsave(plot_path, p, width = plot_width, height = plot_height, dpi = dpi)
cat("Boxplot saved to:", plot_path, "\n")


