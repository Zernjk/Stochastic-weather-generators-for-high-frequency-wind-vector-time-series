# fit_observed.R
#
# Reproduces Table 5 of the paper (Stochastic volatility section): quantile
# regression of |s_{t+1} - s_t| on increasingly rich predictors built from the
# previous ten wind vectors, for tau in {0.5, 0.75, 0.9}.
#
# Input: june_wind.csv (1-min ARM Lamont surface met data, training subset).
# Output: a tidy CSV with one row per (regressors, tau) cell.
#
# Filters applied (same as Stein's original analysis script):
#   - training years 1998-2020
#   - nighttime (minute of day in [120, 720), i.e., 2:00-11:59 UTC) for both
#     the first and last minute of the 11-minute window
#   - eleven consecutive observed minutes (no gaps) ending at t+1
#   - s_t-9 > 1 m/s
#
# Usage:
#   Rscript fit_observed.R <path/to/june_wind.csv> <out_csv>
#
# Requires: quantreg, stringr

suppressPackageStartupMessages({
  library(quantreg)
  library(stringr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript fit_observed.R <input_csv> <output_csv>")
}
in_path  <- args[[1]]
out_path <- args[[2]]

# --- Load and trim to training subset (years 1998-2020) ------------------
W <- read.csv(in_path, header = TRUE)
W <- W[!is.na(W[, 3]), ]

# Stein's training slice maps to the row range below.
# Columns: time, atmos_pressure, temp_mean, rh_mean, wspd_vec_mean,
#          wdir_vec_mean, Easterly, Northerly, year, minute_of_day, set, time_of_day
W <- W[(4 * 1440 * 21 - 1745):(27 * 1440 * 21 - 10925), ]

# --- Time variables from the timestamp -----------------------------------
y  <- as.numeric(noquote(unlist(lapply(W[, 1], substr, start = 1, stop = 4))))
dd <- as.numeric(str_remove(noquote(unlist(lapply(W[, 1], substr, start = 9, stop = 10))), "^0+"))
h  <- as.numeric(str_remove(noquote(unlist(lapply(W[, 1], substr, start = 12, stop = 13))), "^0+"))
m  <- as.numeric(str_remove(noquote(unlist(lapply(W[, 1], substr, start = 15, stop = 16))), "^0+"))
h[is.na(h)] <- 0
m[is.na(m)] <- 0

mc <- 60 * h + m                   # minute of day
my <- mc + 1440 * (dd - 1)         # minute of year starting June 1
night <- (mc >= 120 & mc < 720)    # 2:00-11:59 UTC

# --- Continuity mask: Icon10[i] = TRUE iff minutes i, i+1, ..., i+10 all present
N <- nrow(W)
I <- seq_len(N)
Icon10 <- rep(TRUE, N)
for (yy in unique(y)) {
  idx <- which(y == yy)
  myy <- my[idx]
  nyy <- rep(TRUE, length(myy))
  for (j in 1:10) {
    ttt <- diff(myy, lag = j)
    nyy[c(ttt, rep(0, j)) != j] <- FALSE
  }
  Icon10[idx] <- nyy
}

# --- Index set: continuous 11-minute window, both ends at night, s_t-9 > 1
night_end <- c(night[11:N], rep(FALSE, 10))
n_idx <- I[Icon10 & night_end & night & W[, 5] >= 1]
cat(sprintf("Number of regression rows: %d\n", length(n_idx)))

# --- Build regressors ----------------------------------------------------
# "Current time" t in the paper notation corresponds to n_idx + 9.
ws    <- W[, 5]              # wind speed (wspd_vec_mean)
ue    <- W[, 7]              # easterly
vn    <- W[, 8]              # northerly

y_resp <- abs(ws[n_idx + 10] - ws[n_idx + 9])
s_t    <- ws[n_idx + 9]
s_tm1  <- ws[n_idx + 8]
abs_ds_lag <- function(k) abs(ws[n_idx + 9 - k] - ws[n_idx + 8 - k])
norm_dw_lag <- function(k) sqrt((ue[n_idx + 9 - k] - ue[n_idx + 8 - k])^2 +
                                (vn[n_idx + 9 - k] - vn[n_idx + 8 - k])^2)

abs_ds_full  <- sapply(0:8, abs_ds_lag)
norm_dw_full <- sapply(0:8, norm_dw_lag)

regressor_sets <- list(
  intercept            = NULL,
  st                   = data.frame(s_t),
  st_stm1              = data.frame(s_t, s_tm1),
  st_absds1            = data.frame(s_t, absds1 = abs_ds_lag(0)),
  st_normdw1           = data.frame(s_t, normdw1 = norm_dw_lag(0)),
  st_full_scalar       = data.frame(s_t, abs_ds_full),
  st_full_vector       = data.frame(s_t, norm_dw_full),
  full_vector_no_st    = data.frame(norm_dw_full)
)

# --- Fit one quantile regression -----------------------------------------
fit_one <- function(X, tau) {
  if (is.null(X)) {
    rq(y_resp ~ 1, tau = tau, method = "fn")$rho
  } else {
    rq(y_resp ~ ., data = X, tau = tau, method = "fn")$rho
  }
}

taus <- c(0.5, 0.75, 0.9)
out <- expand.grid(regressors = names(regressor_sets), tau = taus,
                   KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
out$C_tau <- NA_real_

for (i in seq_len(nrow(out))) {
  rn  <- out$regressors[i]
  tau <- out$tau[i]
  out$C_tau[i] <- fit_one(regressor_sets[[rn]], tau)
  cat(sprintf("  %-22s tau=%.2f  C_tau=%.2f\n", rn, tau, out$C_tau[i]))
}

# --- Percent reduction vs intercept-only at each tau ---------------------
for (tau in taus) {
  C0 <- out$C_tau[out$regressors == "intercept" & out$tau == tau]
  mask <- out$tau == tau
  out$pct_reduction[mask] <- 100 * (1 - out$C_tau[mask] / C0)
}

out <- out[, c("regressors", "tau", "C_tau", "pct_reduction")]
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("Wrote %s\n", out_path))
