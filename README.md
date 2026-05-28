# Stochastic Weather Generators for High-Frequency Wind Vector Time Series

This repository contains code, processed data products, pretrained model
artifacts, and reproduction materials for the paper **"Stochastic weather
generators for high-frequency wind vector time series"**.

The repository is organized into four component folders. Each component has its
own `README.md` with detailed instructions, software requirements, and mappings
between files and paper artifacts. This top-level README gives an overview of
the full release and points readers to the appropriate component workflow.

## Repository Structure

| Folder | Purpose |
| --- | --- |
| [`stochastic-wind-vector-replication-code/`](stochastic-wind-vector-replication-code/) | R replication code for descriptive analysis, imputation, no-memorization checks, and energy-score comparisons. |
| [`TimeVQVAE/`](TimeVQVAE/) | Python implementation of the TimeVQVAE-based stochastic wind-vector simulator, including training scripts, checkpoints, notebooks, and generated examples. |
| [`Stochastic Volatility and Quantile Regression/`](<Stochastic Volatility and Quantile Regression/>) | R/Python quantile-regression diagnostics for the stochastic-volatility analysis. |
| [`discriminative_eval/`](discriminative_eval/) | Python binary LSTM discriminator experiments comparing real and synthetic wind time series. |

## Quick Start

1. Clone or download this repository.
2. Choose the component that corresponds to the paper artifact or analysis you
   want to reproduce.
3. Open that component's README and follow its environment setup and run
   commands.

The components are intentionally separated because they use different
dependencies and serve different parts of the paper. In most cases, the safest
workflow is to create a separate R or Python environment inside the component
folder you plan to run.

## Component Overview

### `stochastic-wind-vector-replication-code/`

Final R replication code and supporting data products for reproducing the main
descriptive, imputation, no-memorization, and energy-score analyses.

This component covers:

- Figure 1;
- Figures 12 and 13;
- Figures A1, B1, B2, and F1;
- Tables 2 and F1;
- the zero-wind analysis in Appendix A1.

It uses high-frequency wind-vector and surface meteorology data from the ARM
Southern Great Plains Lamont, Oklahoma E13 site. The data span observations from
`1993-07-21 00:00:00` through `2025-09-20 00:00:00`.

The main workflow is a numbered R-script pipeline:

```r
source("scripts/01_Figure01_Seasonality_and_Zeros.R")
source("scripts/02_Imputation_and_Table2.R")
source("scripts/03_No_Memorization.R")
source("scripts/04_Figures12_and_13_Energy_Scores.R")
```

The repository starts from the included combined Lamont E13 CSV. The
`00_Combining_ARM_data.R` script is optional and only needed to re-download and
recombine the original daily ARM NetCDF/CDF files from a user-specific ARM
order. See
[`stochastic-wind-vector-replication-code/README.md`](stochastic-wind-vector-replication-code/README.md)
for the full data layout, R package requirements, and output directories.

### `TimeVQVAE/`

Python implementation of the TimeVQVAE-based stochastic wind simulator. The
model uses a two-stage pipeline:

1. Stage 1 trains a VQ-VAE to learn a discrete vector-quantized representation
   of wind time-series windows.
2. Stage 2 trains a MaskGIT-style transformer prior over the learned tokens to
   generate new wind trajectories.

This implementation is adapted from the TimeVQVAE model and codebase by Lee,
Malacarne, and Aune for vector-quantized time-series generation. The wind
simulation version adds support for weather-state information, weather-state
conditioning/embedding variants, consecutive multi-day generation, and
weather-state masking experiments.

The folder includes:

- prepared NumPy datasets in `datasets/`;
- `datasets/data preprocess.ipynb`, which documents how the prepared `.npz`
  datasets are created from minute-level wind-component CSV data;
- YAML configuration files in `configs/`;
- training entry points `stage1.py` and `stage2.py`;
- pretrained checkpoints in `saved_models/`;
- notebooks for simulation workflows;
- example synthetic outputs in `output_synthetic/`.

See [`TimeVQVAE/README.md`](TimeVQVAE/README.md) for environment setup,
configuration options, training commands, sampling instructions, and citation
information for the original TimeVQVAE method.

The TimeVQVAE component builds on:

```bibtex
@InProceedings{pmlr-v206-lee23d,
  title = {Vector Quantized Time Series Generation with a Bidirectional Prior Model},
  author = {Lee, Daesoo and Malacarne, Sara and Aune, Erlend},
  booktitle = {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = {7665--7693},
  year = {2023},
  editor = {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = {206},
  series = {Proceedings of Machine Learning Research},
  month = {25--27 Apr},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v206/lee23d.html}
}
```

### `Stochastic Volatility and Quantile Regression/`

Code and data for the quantile-regression diagnostics used in the stochastic
volatility analysis.

This component reproduces:

- Table 5 in Section 7;
- Figure 19 in Section 7;
- Figure D1 in Appendix D;
- Table D1 in Appendix D;
- the raw long-format synthetic `C_tau` values behind those summaries.

It contains two independent pipelines:

- an R pipeline for observational June wind data;
- a Python pipeline for synthetic Weather as Embeddings and Weather as Features
  runs.

The fitted response is the one-minute absolute change in wind speed,
`|s_{t+1} - s_t|`, with quantile regressions fit at tau values `0.5`, `0.75`,
and `0.9` across several scalar and vector lag-history regressor sets.

See
[`Stochastic Volatility and Quantile Regression/README.md`](<Stochastic Volatility and Quantile Regression/README.md>)
for the exact commands, package requirements, data provenance, and mapping from
paper artifacts to files.

### `discriminative_eval/`

Minimal Python code for binary LSTM discriminator experiments comparing
synthetic and real wind time series.

The component includes two discriminator input constructions:

- `one_day/`: one calendar day of 1-minute wind-vector observations with shape
  `(1440, 2)`;
- `three_day/`: three consecutive days concatenated along time with shape
  `(4320, 2)`.

The one-day pipeline can optionally apply Butterworth low-pass or high-pass
filtering through command-line options. Training scripts write logs and
checkpoints under the working directory from which `source.py` is launched.

This folder intentionally does not distribute `const.py`. Users must create a
`const.py` file in both `one_day/` and `three_day/` with the path constants
required by each folder's `dataset.py`.

See [`discriminative_eval/README.md`](discriminative_eval/README.md) for the
folder layout, dependency file, and instructions for defining local path
constants.

## Suggested Reading Order

1. Start with
   [`stochastic-wind-vector-replication-code/README.md`](stochastic-wind-vector-replication-code/README.md)
   to understand the observed ARM data, processed training/testing sets,
   synthetic data products, and paper-level replication outputs.
2. Read [`TimeVQVAE/README.md`](TimeVQVAE/README.md) for the generative-model
   training and simulation workflow.
3. Read
   [`Stochastic Volatility and Quantile Regression/README.md`](<Stochastic Volatility and Quantile Regression/README.md>)
   for the stochastic-volatility diagnostics.
4. Read [`discriminative_eval/README.md`](discriminative_eval/README.md) for
   the LSTM discriminative evaluation setup.

## Reproducibility Scope

This repository is intended to support reproduction of the analyses and
simulation workflows described in the paper. The included materials cover:

- preprocessing and analysis of ARM Lamont E13 wind-vector and surface
  meteorology data;
- TimeVQVAE model training and sampling workflows for stochastic wind-vector
  generation;
- energy-score, no-memorization, stochastic-volatility, and discriminative
  evaluation analyses;
- processed outputs and example figures/tables documented in the component
  READMEs.

Some workflows depend on user-specific local paths or external data-order
metadata. Those cases are documented in the relevant component README.

## Software Requirements

The full repository uses both R and Python. Dependencies are managed separately
in the component folders.

- The replication-code folder uses R scripts and installs required R packages
  from within the scripts if they are missing.
- The TimeVQVAE folder uses Python with PyTorch, PyTorch Lightning, Weights &
  Biases, and the packages listed in its `requirements.txt`.
- The quantile-regression diagnostics use R packages including `quantreg` and
  `stringr`, plus Python packages including `pandas`, `numpy`, and
  `statsmodels`.
- The discriminative-evaluation folder provides a minimal Python
  `requirements.txt`.

Because the components serve different parts of the paper, create environments
within each component folder and follow that folder's README before running its
scripts.

## Data Notes

The repository includes processed data products and synthetic outputs used by
the paper workflows. The original daily ARM NetCDF/CDF files and ARM
download-session files are not included because they are tied to an individual
ARM data order and browser session. Users who want to re-download the original
ARM files should create their own ARM order and provide their own catalog and
cookies files as described in the [`stochastic-wind-vector-replication-code/README.md`](stochastic-wind-vector-replication-code/README.md)

Local path configuration files used for the discriminative evaluation are also
not included. They must be created by each user for their own machine.

## Citation

If using this repository, please cite the accompanying paper:

```text
Stochastic weather generators for high-frequency wind vector time series.
```
- Software DOI: [10.5281/zenodo.20421181](https://doi.org/10.5281/zenodo.20421181)
- Dataset DOI: [10.5281/zenodo.20421237](https://doi.org/10.5281/zenodo.20421237)
