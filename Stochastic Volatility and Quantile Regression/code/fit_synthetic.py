"""
fit_synthetic.py

Quantile-regression diagnostics applied to each synthetic wind dataset.
This is the pipeline used to populate Figures 19 and D1 and Table D1 of the
paper for the five Weather-as-Embeddings and five Weather-as-Features runs.

Each input CSV is a two-column file: (Easterly, Northerly) wind components,
1 row per minute, no header.

For each input file the script fits quantile regressions of |s_{t+1} - s_t|
against the same eight regressor sets used in Table 5 of the paper, at
tau in {0.5, 0.75, 0.9}, and writes one row per (file, regressors, tau) to
the output CSV.

Filters: nighttime (minute of day in [120, 720) UTC) for both t and t+1,
and s_t > 1 m/s.  No continuity check is applied: the synthetic series are
sampled day-by-day with each day internally continuous, and day boundaries
are treated as continuous in time (which matches how the generators were
trained).  This differs slightly from the filter used by fit_observed.R on
the observational record; see README.md for a discussion.

Usage:
    python fit_synthetic.py <data_dir> <out_csv>

<data_dir> should contain two subdirectories named "synthetic_embedded" and
"synthetic_features", each holding run_1.csv through run_5.csv.

Requires: pandas, numpy, statsmodels (>= 0.13).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def pinball(y, X, beta, tau):
    """Minimized check-loss criterion C_tau."""
    r = y - X.dot(beta)
    return float(np.where(r >= 0, tau * r, (tau - 1) * r).sum())


def fit_one_dataset(u, v, taus=(0.5, 0.75, 0.9), lags=8):
    """Return a dict {(regressors, tau): C_tau} for one (u, v) series."""
    s = np.sqrt(u ** 2 + v ** 2)
    N = len(s)

    t_candidates = np.arange(lags + 1, N - 1)
    minute_in_day = np.arange(N) % 1440
    night = (minute_in_day >= 120) & (minute_in_day < 720)
    mask = night[t_candidates] & night[t_candidates + 1] & (s[t_candidates] > 1.0)
    t = t_candidates[mask]
    if len(t) < 100:
        raise ValueError(f"Too few rows after filter: {len(t)}")

    y = np.abs(s[t + 1] - s[t])
    st = s[t]
    st_m1 = s[t - 1]
    absds1 = np.abs(st - st_m1)

    lag_ds = np.zeros((len(t), lags + 1))
    lag_dw = np.zeros((len(t), lags + 1))
    for k in range(lags + 1):
        idx = t - k
        prev = idx - 1
        lag_ds[:, k] = np.abs(s[idx] - s[prev])
        lag_dw[:, k] = np.sqrt((u[idx] - u[prev]) ** 2 + (v[idx] - v[prev]) ** 2)
    dwnorm1 = lag_dw[:, 0]

    ones = np.ones((len(t), 1))
    models = {
        "intercept":         ones,
        "st":                np.column_stack([ones, st]),
        "st_stm1":           np.column_stack([ones, st, st_m1]),
        "st_absds1":         np.column_stack([ones, st, absds1]),
        "st_normdw1":        np.column_stack([ones, st, dwnorm1]),
        "st_full_scalar":    np.column_stack([ones, st, lag_ds]),
        "st_full_vector":    np.column_stack([ones, st, lag_dw]),
        "full_vector_no_st": np.column_stack([ones, lag_dw]),
    }

    out = {}
    for tau in taus:
        for name, X in models.items():
            res = sm.QuantReg(y, X).fit(q=float(tau), max_iter=5000)
            out[(name, float(tau))] = pinball(y, X, res.params, float(tau))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Folder containing synthetic_embedded/ and synthetic_features/")
    ap.add_argument("out_csv",  help="Output CSV (long format)")
    args = ap.parse_args()

    rows = []
    data_dir = Path(args.data_dir)
    for generator in ("embedded", "features"):
        sub = data_dir / f"synthetic_{generator}"
        for run in range(1, 6):
            path = sub / f"run_{run}.csv"
            print(f"Fitting {generator} run {run}: {path}")
            df = pd.read_csv(path, header=None)
            u = df.iloc[:, 0].values.astype(float)
            v = df.iloc[:, 1].values.astype(float)
            results = fit_one_dataset(u, v)
            for (name, tau), C in results.items():
                rows.append({
                    "generator":  generator,
                    "run":        run,
                    "regressors": name,
                    "tau":        tau,
                    "C_tau":      C,
                })

    res = pd.DataFrame(rows)
    # Add percent reduction vs intercept-only within each (generator, run, tau).
    res["pct_reduction"] = np.nan
    for (g, r, tau), sub in res.groupby(["generator", "run", "tau"]):
        C0 = sub.loc[sub["regressors"] == "intercept", "C_tau"].iloc[0]
        idx = sub.index
        res.loc[idx, "pct_reduction"] = 100.0 * (1.0 - res.loc[idx, "C_tau"] / C0)

    res = res[["generator", "run", "regressors", "tau", "C_tau", "pct_reduction"]]
    res.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(res)} rows)")


if __name__ == "__main__":
    main()
