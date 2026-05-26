# ============================================================
# Wind Data — Zero Analysis & Seasonality Plot
# ============================================================
#
# WHAT THIS SCRIPT PRODUCES:
#   1. A CSV summarizing zero wind speed blocks in June
#   2. A PDF with median and IQR diurnal wind speed plots,
#      one line per selected month, averaged into 10-minute bins.
#
# ============================================================
# Edit only this section
# ============================================================

# --- Input data ---
data_file     = "data/LamontOK_E13_19930721_20250920.csv"

# --- Output folder ---
output_dir    = "Seasonality and Zero Analysis"

# --- Zero analysis settings ---
zero_month          = 6
zero_month_name     = "June"        # Used in output file names and column labels

# Year range to include in zero analysis (inclusive)
zero_year_start     = 1994
zero_year_end       = 2025

# Variables that define a "suspicious" missing-data coincidence
suspect_vars        = c("temp_mean", "rh_mean")   # Both must be NA to flag a row

# Wind speed variable to examine for zeros
wind_var            = "wspd_vec_mean"

# --- Seasonality plot settings ---
# Months to include: named vector where names are display labels
# and values are the corresponding month numbers
plot_months = c(
  "March"     = 3,
  "June"      = 6,
  "September" = 9,
  "December"  = 12
)

# Colors for each month label (names must match names of plot_months)
month_colors = c(
  "March"     = "steelblue",
  "June"      = "black",
  "September" = "orange",
  "December"  = "red"
)

# Year range to include in seasonality plot (inclusive)
plot_year_start = 1998
plot_year_end   = 2020

# Output plot file name (saved inside output_dir)
plot_filename   = "fig01.pdf"
plot_width      = 10      # inches
plot_height     = 5.5     # inches

# ============================================================
# Load/download needed packages
# ============================================================

packages = c("ggplot2", "grid", "dplyr", "tidyr", "lubridate",
             "gridExtra", "readr", "data.table")

installed = packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])
invisible(lapply(packages, library, character.only = TRUE))

# ============================================================
# No need to edit below this line
# ============================================================

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

zero_csv = file.path(
  output_dir,
  paste0("zero_analysis_", tolower(zero_month_name), ".csv")
)

plot_path = file.path(output_dir, plot_filename)

# ============================================================
# Load data
# ============================================================

wind = readr::read_csv(data_file) %>%
  dplyr::mutate(
    time          = as.POSIXct(time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"),
    year          = lubridate::year(time),
    month         = lubridate::month(time),
    minute_of_day = (lubridate::hour(time) * 60 + lubridate::minute(time)) + 1
  )

# ============================================================
# Zero analysis
# ============================================================

# --- Subset to target month and year range ---
june = wind %>%
  dplyr::filter(
    month == zero_month,
    year  >= zero_year_start,
    year  <= zero_year_end
  )

# --- All zero wind speed rows ---
zeros = june %>%
  dplyr::filter(.data[[wind_var]] == 0)

# --- Rows where ALL suspect variables are missing ---
suspect_filter = Reduce(`&`, lapply(suspect_vars, function(v) is.na(june[[v]])))
sus = june[suspect_filter, ]

# --- Overlap: suspect rows that also have zero wind speed ---
sus_zero_overlap = sus %>%
  dplyr::filter(.data[[wind_var]] == 0)

# --- Exception: suspect rows where wind speed is NOT zero ---
sus_nonzero = sus %>%
  dplyr::filter(!is.na(.data[[wind_var]]) & .data[[wind_var]] != 0)


# --- Summary counts (printed to console) ---
cat("\n=== ZERO ANALYSIS SUMMARY ===\n")
cat("Month examined        :", zero_month_name, "\n")
cat("Year range            :", zero_year_start, "to", zero_year_end, "\n")
cat("Total zero wind rows  :", nrow(zeros), "\n")
cat("Rows with", paste(suspect_vars, collapse = " & "), "missing:", nrow(sus), "\n")
cat("Of those — wind also 0:", nrow(sus_zero_overlap), "\n")
cat("Of those — wind NOT 0 :", nrow(sus_nonzero), "\n")
if (nrow(sus_nonzero) > 0) {
  cat("  Exceptions:\n")
  print(sus_nonzero %>% dplyr::select(time, all_of(c(wind_var, suspect_vars))))
}

# --- Block-level summary with context wind speeds ---

# Add a row index to the full June data for neighbor lookups
june_indexed = june %>%
  dplyr::arrange(time) %>%
  dplyr::mutate(.row_idx = dplyr::row_number())

# Identify zero rows by index
zero_indices = june_indexed %>%
  dplyr::filter(.data[[wind_var]] == 0) %>%
  dplyr::pull(.row_idx)

# Group contiguous zero rows into blocks
zero_blocks = june_indexed %>%
  dplyr::filter(.data[[wind_var]] == 0) %>%
  dplyr::arrange(time) %>%
  dplyr::mutate(
    new_block = is.na(dplyr::lag(time)) |
      as.numeric(difftime(time, dplyr::lag(time), units = "mins")) > 1,
    block_id  = cumsum(new_block)
  )

# For each block, find the wind speed immediately before and after
block_context = zero_blocks %>%
  dplyr::group_by(block_id) %>%
  dplyr::summarise(
    date              = as.Date(min(time)),
    start_time        = format(min(time), "%H:%M"),
    end_time          = format(max(time), "%H:%M"),
    n_minutes         = dplyr::n(),
    # Are ALL rows in the block missing the suspect variables?
    all_suspect_na    = all(Reduce(`&`, lapply(suspect_vars,
                                               function(v) is.na(dplyr::cur_data()[[v]])))),
    first_block_idx   = min(.row_idx),
    last_block_idx    = max(.row_idx),
    .groups           = "drop"
  ) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    # Look up the wind speed one row before the block starts
    wind_before = {
      idx = first_block_idx - 1
      if (idx >= 1) june_indexed[[wind_var]][june_indexed$.row_idx == idx][1] else NA_real_
    },
    # Look up the wind speed one row after the block ends
    wind_after  = {
      idx = last_block_idx + 1
      if (idx <= nrow(june_indexed)) june_indexed[[wind_var]][june_indexed$.row_idx == idx][1] else NA_real_
    }
  ) %>%
  dplyr::ungroup() %>%
  dplyr::select(
    date, start_time, end_time, n_minutes,
    all_suspect_na, wind_before, wind_after
  ) %>%
  dplyr::rename(
    Date                              = date,
    `Start Time`                      = start_time,
    `End Time`                        = end_time,
    `Minutes`                         = n_minutes,
    `All Suspect Vars Missing`        = all_suspect_na,
    `Wind Speed Before Block (m/s)`   = wind_before,
    `Wind Speed After Block (m/s)`    = wind_after
  ) %>%
  dplyr::arrange(Date, `Start Time`)

# Save the block-level table
write_csv(block_context, zero_csv)
cat("Zero analysis table saved to:", zero_csv, "\n\n")
print(block_context, n = Inf)

# ============================================================
# Seasonality Plot (Figure 01)
# ============================================================

month_nums  = unname(plot_months)
month_names = names(plot_months)

filtered_wind = wind %>%
  dplyr::filter(
    month %in% month_nums,
    year  >= plot_year_start,
    year  <= plot_year_end
  ) %>%
  dplyr::mutate(
    month_name = factor(month, levels = month_nums, labels = month_names)
  )

# Per-minute statistics by month
diurnal_stats = filtered_wind %>%
  dplyr::group_by(month_name, minute_of_day) %>%
  dplyr::summarise(
    n           = dplyr::n(),
    mean_wspd   = mean(.data[[wind_var]], na.rm = TRUE),
    median_wspd = median(.data[[wind_var]], na.rm = TRUE),
    iqr_wspd    = IQR(.data[[wind_var]], na.rm = TRUE),
    .groups     = "drop"
  )

# Collapse into 10-minute bins
diurnal_stats_10min = diurnal_stats %>%
  dplyr::mutate(ten_min_bin = 10 * ceiling(minute_of_day / 10)) %>%
  dplyr::group_by(month_name, ten_min_bin) %>%
  dplyr::summarise(
    mean_wspd_10min = mean(mean_wspd,   na.rm = TRUE),
    mean_med_10min  = mean(median_wspd, na.rm = TRUE),
    mean_iqr_10min  = mean(iqr_wspd,    na.rm = TRUE),
    .groups         = "drop"
  ) %>%
  dplyr::mutate(bin_center = ten_min_bin - 4.5)

# Shared theme
custom_theme = theme_minimal() +
  theme(
    plot.title       = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10),
    legend.position  = "bottom",
    panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
    panel.grid.minor = element_line(color = "gray95", linewidth = 0.25),
    legend.title     = element_text(size = 12),
    legend.text      = element_text(size = 10)
  )

# Helper to extract shared legend
get_legend = function(p) {
  tmp = ggplot_gtable(ggplot_build(p))
  leg = which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  tmp$grobs[[leg]]
}

legend_plot = ggplot(diurnal_stats_10min,
                     aes(x = bin_center / 60, y = mean_wspd_10min, color = month_name)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = month_colors) +
  labs(color = "Month") +
  theme(
    legend.position  = "bottom",
    legend.title     = element_text(size = 14),
    legend.text      = element_text(size = 12),
    legend.key.size  = unit(1.5, "cm"),
    legend.background = element_rect(fill = "transparent"),
    legend.key        = element_rect(fill = "transparent")
  )

shared_legend = get_legend(legend_plot)

med_plot = ggplot(diurnal_stats_10min,
                  aes(x = bin_center / 60, y = mean_med_10min, color = month_name)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = month_colors) +
  scale_x_continuous(breaks = c(0, 4, 8, 12, 16, 20, 24)) +
  labs(title = "Median", y = "Wind Speed (m/s)", x = "Hour") +
  custom_theme +
  theme(legend.position = "none")

iqr_plot = ggplot(diurnal_stats_10min,
                  aes(x = bin_center / 60, y = mean_iqr_10min, color = month_name)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = month_colors) +
  scale_x_continuous(breaks = c(0, 4, 8, 12, 16, 20, 24)) +
  labs(title = "IQR", y = "Wind Speed (m/s)", x = "Hour") +
  custom_theme +
  theme(legend.position = "none")

combined_plot = gridExtra::grid.arrange(
  gridExtra::arrangeGrob(med_plot, iqr_plot, ncol = 2),
  shared_legend,
  nrow    = 2,
  heights = c(10, 2)
)

ggsave(plot_path, combined_plot, width = plot_width, height = plot_height, dpi = 300)
cat("Seasonality plot saved to:", plot_path, "\n")

