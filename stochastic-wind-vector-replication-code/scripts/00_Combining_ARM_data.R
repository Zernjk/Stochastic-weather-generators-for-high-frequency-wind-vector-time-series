# The exact order used for the paper was 1993-07-21 00:00:00 to 2025-09-20 00:00:00
# https://adc.arm.gov/discovery/#/results/id::sgpmetE13.b1_atmos_pressure_sfcmet_met_sfcmet?dataLevel=b1&showDetails=true

# ============================================================
# ARM Surface Meteorology Data Downloading & Combining
# ============================================================
#
# HOW TO ORDER DATA FROM ARM:
#   1. Go to: https://adc.arm.gov/discovery/
#   2. Search for desired data (e.g. sgpmetE13.b1)
#   3. Select date range and submit order
#   4. ARM will email you a THREDDS link when ready
#   5. Open the THREDDS link in your browser and log in
#   6. Right-click the data folder -> "Save Link As" -> save as catalog.html
#   7. Use the browser extension "Get cookies.txt LOCALLY" to export
#      cookies while THREDDS is open in your browser
#
# ============================================================
# Edit only this section
# ============================================================

# --- ARM order details ---
order_number  = "266087"                  # Found in your THREDDS URL, ARM should also tell you
datastream    = "sgpmetE13.b1"            # ARM datastream identifier

# --- Input files (place in working directory or provide full paths) ---
catalog_html  = "catalog.html"
cookie_file   = "thredds-ui.svcs.arm.gov_cookies.txt"

# --- Output naming ---
site_name     = "LamontOK"               # Site labe/location
facility_id   = "E13"                    # ARM facility code
date_start    = "19930721"               # Start date (YYYYMMDD)
date_end      = "20250920"               # End date   (YYYYMMDD)

# --- Folder structure (all relative to working directory) ---
data_subdir   = "data"                                 # Folder for all output data
raw_subdir    = "raw files"                            # Subfolder for raw NetCDF files

# ============================================================
# No need to edit below this line
# ============================================================

base_dir  = paste0(
  "https://thredds-ui.svcs.arm.gov/thredds/fileServer/orders/gj1/",
  order_number, "/", datastream, "/"
)

dest_dir  = file.path(data_subdir, raw_subdir)

output_csv = file.path(
  data_subdir,
  paste0(site_name, "_", facility_id, "_", date_start, "_", date_end, ".csv")
)

# ============================================================
# Load/download needed packages
# ============================================================

packages = c("rvest", "stringr", "purrr", "curl", "readr", "dplyr", "ncdf4")

installed = packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

invisible(lapply(packages, library, character.only = TRUE))

# ============================================================
# Initial setup
# ============================================================

dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot(file.exists(catalog_html), file.exists(cookie_file))

# Curl handle with cookies and redirects
h = new_handle()
handle_setheaders(h,
                  "User-Agent" = "R (curl)",
                  "Accept"     = "application/x-netcdf, application/octet-stream, */*",
                  "Referer"    = "https://thredds-ui.svcs.arm.gov/")
handle_setopt(h,
              followlocation = TRUE,
              cookiefile     = normalizePath(cookie_file),
              cookiejar      = tempfile("cookiejar_"))

# ============================================================
# Download CDF Files from THREDDS
# ============================================================

# Parse catalog for NetCDF links
pg       = read_html(catalog_html)
hrefs    = html_attr(html_elements(pg, "a"), "href") |> na.omit()
nc_links = unique(grep("\\.(nc|cdf)($|\\?)", hrefs, ignore.case = TRUE, value = TRUE))

# Normalize to full URLs
urls = ifelse(grepl("^https?://", nc_links), nc_links,
              paste0(base_dir, basename(nc_links)))
urls = sub("^http://", "https://", urls)

cat("Found", length(urls), "NetCDF URLs\n"); stopifnot(length(urls) > 0)

# Detect if a saved file is actually an HTML login/error page
looks_like_html = function(path) {
  if (!file.exists(path)) return(TRUE)
  return(file.info(path)$size < 10000)
}

# Download one file with retries
dl_one = function(u, handle, tries = 3, pause = 0.3) {
  dest = file.path(dest_dir, basename(u))
  for (i in seq_len(tries)) {
    ok = try({ curl_download(u, destfile = dest, mode = "wb", handle = handle); TRUE }, silent = TRUE)
    if (isTRUE(ok) && file.exists(dest) && file.info(dest)$size > 0) {
      if (!looks_like_html(dest)) return(TRUE)
      html_dest = paste0(dest, ".html")
      if (file.exists(html_dest)) unlink(html_dest)
      file.rename(dest, html_dest)
    }
    Sys.sleep(pause)
  }
  message("Failed: ", u); FALSE
}

ok = vapply(urls, dl_one, logical(1), handle = h)
cat("Succeeded:", sum(ok), " | Failed:", sum(!ok), "\n")

if (any(!ok)) writeLines(urls[!ok], file.path(dest_dir, "_failed_urls.txt"))

# ============================================================
# Read and combine NetCDF files
# ============================================================

nc_files = list.files(dest_dir, pattern = "\\.(nc|cdf)$", full.names = TRUE)

read_netcdf_data = function(file) {
  tryCatch({
    nc = nc_open(file)
    
    base_time_vec  = as.vector(ncvar_get(nc, "base_time"))
    time_offset_vec = as.vector(ncvar_get(nc, "time_offset"))
    time_values    = as.POSIXct(base_time_vec + time_offset_vec,
                                origin = "1970-01-01", tz = "UTC")
    
    data_list = list(
      time        = time_values,
      base_time   = as.POSIXct(base_time_vec, origin = "1970-01-01", tz = "UTC"),
      time_offset = as.numeric(time_offset_vec)
    )
    
    # Explicitly name the desired variables. Recent years have additional variables that
    # were not measured earlier, so combining blindly will break down
    variables_to_extract = c(
      "atmos_pressure", "qc_atmos_pressure",
      "temp_mean", "qc_temp_mean", "temp_std",
      "rh_mean", "qc_rh_mean", "rh_std",
      "vapor_pressure_mean", "qc_vapor_pressure_mean", "vapor_pressure_std",
      "wspd_arith_mean", "qc_wspd_arith_mean",
      "wspd_vec_mean", "qc_wspd_vec_mean",
      "wdir_vec_mean", "qc_wdir_vec_mean", "wdir_vec_std",
      "tbrg_precip_total", "qc_tbrg_precip_total",
      "tbrg_precip_total_corr", "qc_tbrg_precip_total_corr",
      "logger_volt", "qc_logger_volt",
      "logger_temp", "qc_logger_temp",
      "lat", "lon", "alt"
    )
    
    for (var_name in variables_to_extract) {
      if (var_name %in% names(nc$var)) {
        data_list[[var_name]] = as.vector(ncvar_get(nc, var_name))
      } else {
        data_list[[var_name]] = rep(NA, length(time_values))
      }
    }
    
    nc_close(nc)
    as_tibble(data_list)
    
  }, error = function(e) {
    warning("Failed to read: ", basename(file), " - ", e$message)
    return(tibble())
  })
}

combined_data = nc_files %>%
  lapply(read_netcdf_data) %>%
  bind_rows()

glimpse(combined_data)

# ============================================================
# SAVE
# ============================================================

write_csv(combined_data, output_csv)
cat("Saved to:", output_csv, "\n")