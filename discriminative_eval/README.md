# Discriminative evaluation code

This package contains minimal code to reproduce the **binary LSTM discriminator** experiments on **synthetic vs. real** wind time series under two input constructions:

- **`one_day/`** — each sample is **one calendar day** of 1-minute resolution wind (shape **(1440, 2)**: Easterly, Northerly). Optional **Butterworth low-pass / high-pass** filtering (cycles/day cutoff) is applied in the dataset pipeline when enabled via CLI.
- **`three_day/`** — each sample is **three consecutive days** concatenated along time (shape **(4320, 2)**), built by sliding windows over multi-day batches (see `dataset.py` for details). This variant matches the “three-day window” discriminator setup in the paper supplement / methods (as applicable).

Training scripts write logs and checkpoints under the working directory (e.g. `log/`, `model_checkpoints/`) relative to where you launch `source.py`.

---

## What is **not** included: `const.py`

**`const.py` is intentionally not distributed** (e.g. data use agreements, paths on the HPC cluster, or journal anonymity). You must create a file named **`const.py`** in **`one_day/`** and **`three_day/`** respectively (same variable names; paths may differ per machine).

Each folder’s `dataset.py` imports path constants from `const`. Define **at least** the symbols imported in that folder’s `dataset.py`:

---

## Folder layout

```text
discriminative_eval/
├── README.md                 # this file
├── requirements.txt          # minimal pip dependencies
├── one_day/
│   ├── dataset.py            # 1-day samples; optional spectral filter
│   ├── model.py              # WindModelLSTM
│   ├── source.py             # training CLI
│   └── run.sh                # example batch launcher (edit paths / GPU)
└── three_day/
    ├── dataset.py            # 3-day sliding windows (4320 points)
    ├── model.py              # same architecture as one_day (mean-pool over time)
    ├── source.py
    └── run.sh                # example batch launcher