# Quantile-regression diagnostics for stochastic wind generators

Code and data to reproduce the quantile-regression analysis in Section 7
(Stochastic volatility) and Appendix D (Additional quantile-regression
diagnostics) of the paper.

## What's in the package

```
qr_submission/
├── README.md                          (this file)
├── code/
│   ├── fit_observed.R                 R script for the observational data
│   ├── fit_synthetic.py               Python script for the synthetic runs
│   ├── plot_figure_19.py              builds Figure 19 (main paper)
│   └── plot_figure_D1.py              builds Figure D1 (appendix)
├── data/
│   ├── observed/
│   │   └── june_wind.csv              minute-level ARM Lamont surface met
│   ├── synthetic_embedded/
│   │   └── run_1.csv ... run_5.csv    5 runs of Weather as Embeddings
│   └── synthetic_features/
│       └── run_1.csv ... run_5.csv    5 runs of Weather as Features
├── results/
│   ├── observed_table5.csv            Table 5 of the paper
│   ├── synthetic_raw_C_tau.csv        240 rows: every synthetic C_tau
│   └── table_D1.csv                   Table D1 mean (SD) summary
└── example_figures/
    ├── figure_19.{pdf,png}
    └── figure_D1.{pdf,png}
```

## Mapping from artifacts to paper

| Paper artifact              | File in this package                      |
|-----------------------------|-------------------------------------------|
| Table 5 (Section 7)         | `results/observed_table5.csv`             |
| Figure 19 (Section 7)       | `example_figures/figure_19.{pdf,png}`     |
| Figure D1 (Appendix D)      | `example_figures/figure_D1.{pdf,png}`     |
| Table D1 (Appendix D)       | `results/table_D1.csv`                    |
| Raw synthetic C_tau values  | `results/synthetic_raw_C_tau.csv`         |

`synthetic_raw_C_tau.csv` is the full long-format table behind both
Figure 19 (panels a and b) and Figure D1 — 240 rows = 2 generators × 5 runs ×
8 regressor sets × 3 tau values.

## How to reproduce

Two independent pipelines, one per data source.

### 1. Observational data (R)

```
cd code
Rscript fit_observed.R ../data/observed/june_wind.csv ../results/observed_table5.csv
```

This produces a CSV that matches Table 5 of the paper.  Requires the
`quantreg` and `stringr` R packages.

### 2. Synthetic data (Python)

```
cd code
python fit_synthetic.py ../data ../results/synthetic_raw_C_tau.csv
```

This loops over the ten synthetic CSVs (5 Embeddings + 5 Features) and
writes one long-format CSV with `C_tau` and percent reduction for each
(generator, run, regressors, tau).  Requires `pandas`, `numpy`, and
`statsmodels >= 0.13`.

### 3. Figures

```
cd code
python plot_figure_19.py --results_dir ../results --out ../figure_19
python plot_figure_D1.py --results_dir ../results --out ../figure_D1
```

Both plotting scripts read the CSVs in `results/`.  PDFs and PNGs are
written to the location given by `--out`.  The figures in
`example_figures/` were produced with these commands.

## Models fit

Eight regressor sets, each fit at tau in {0.5, 0.75, 0.9}:

| Key in CSVs           | Regressors                                              |
|-----------------------|---------------------------------------------------------|
| `intercept`           | 1                                                       |
| `st`                  | s_t                                                     |
| `st_stm1`             | s_t, s_{t-1}                                            |
| `st_absds1`           | s_t, \|s_t - s_{t-1}\|                                  |
| `st_normdw1`          | s_t, \|\|w_t - w_{t-1}\|\|                              |
| `st_full_scalar`      | s_t, \|s_t - s_{t-1}\|, ..., \|s_{t-8} - s_{t-9}\|      |
| `st_full_vector`      | s_t, \|\|w_t - w_{t-1}\|\|, ..., \|\|w_{t-8} - w_{t-9}\|\| |
| `full_vector_no_st`   | \|\|w_t - w_{t-1}\|\|, ..., \|\|w_{t-8} - w_{t-9}\|\|   |

The response in all cases is `|s_{t+1} - s_t|`, the one-minute absolute
change in wind speed.

The figures use only four of these (`st`, `st_normdw1`, `st_full_scalar`,
`st_full_vector`), labelled "Scalar Lag 1", "Vector Lag 1", "Full Scalar
History", "Full Vector History".

## Note on filter differences between R and Python pipelines

The observational pipeline (R) and the synthetic pipeline (Python) apply
slightly different row filters before fitting.  This was the case in the
analysis as it ran for the paper, and both scripts here preserve that
behaviour.

| Filter                | `fit_observed.R`                | `fit_synthetic.py`           |
|-----------------------|---------------------------------|------------------------------|
| Nighttime check       | both `t-9` and `t+1` at night  | both `t` and `t+1` at night |
| Wind-speed cutoff     | `s_{t-9} > 1` m/s              | `s_t > 1` m/s               |
| Continuity check      | all 11 minutes observed        | none (synthetic series are  |
|                       | (`Icon10`)                     | continuous within each day)  |
| Quantile-reg solver   | R `quantreg` Frisch-Newton     | `statsmodels` IRLS           |

For the synthetic data the continuity check is moot: each day is internally
continuous, and day boundaries are stitched continuously by the generators.
For the observational data the continuity check matters because the
training period has nine fully missing days; the `Icon10` mask excludes
windows that would straddle those gaps.

The two solvers agree to within numerical precision on convex quantile
regression problems, so the choice does not materially affect the results.

## Data provenance

* `data/observed/june_wind.csv` — minute-level surface meteorology from the
  ARM Lamont, Oklahoma facility (Kyrouac & Ritsche, 2025), reformatted as
  documented in Section 2 of the paper.
* `data/synthetic_embedded/run_*.csv` — five independent samples of 23
  years × 21 days × 1440 minutes generated by the Weather as Embeddings
  generator described in Section 4.4 of the paper.
* `data/synthetic_features/run_*.csv` — same, for the Weather as Features
  generator.

## Sanity check

The numbers in `results/synthetic_raw_C_tau.csv` were obtained by running
the script in `code/fit_synthetic.py` on the ten files in
`data/synthetic_*/run_*.csv`.  As a spot check:

| File                          | Intercept C_0.9 (paper) | Computed |
|-------------------------------|-------------------------|----------|
| `synthetic_embedded/run_1.csv`| 23909.47                | 23909.47 |
| `synthetic_features/run_1.csv`| 17056.78                | 17057.29 |

The small drift on the Features side reflects float-precision differences
in how the source xlsx files were written and re-read; the percent
reductions agree to better than 0.5 percentage points everywhere.
