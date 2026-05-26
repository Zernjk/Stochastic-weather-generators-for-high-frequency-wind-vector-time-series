"""
plot_figure_D1.py

Reproduce Figure D1 of the appendix: percent reduction in C_tau as model
complexity grows, at tau = 0.5 (panel a) and tau = 0.75 (panel b).

Reads:
  - results/observed_table5.csv
  - results/synthetic_raw_C_tau.csv

Writes:
  - figure_D1.pdf
  - figure_D1.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_WIDTH, FIG_HEIGHT, FIG_DPI = 8, 3.7, 300
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 10,
    "font.weight": "normal",
    "axes.labelweight": "normal",
})

JITTER_X, SEED = 0.08, 7
COLOR_FEATURES = "#27ae60"
COLOR_EMBEDDED = "#e74c3c"
COLOR_OBS_LINE = "black"
OBS_FACE = (0, 0, 0, 0.75)
OBS_EDGE = (0, 0, 0, 0.55)
SIM_ALPHA = 0.7

MODEL_RUNGS = [
    ("st",             "Scalar\nLag 1"),
    ("st_normdw1",     "Vector\nLag 1"),
    ("st_full_scalar", "Full Scalar\nHistory"),
    ("st_full_vector", "Full Vector\nHistory"),
]


def style_axes(ax):
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.spines["top"].set_linewidth(1.0)
    ax.spines["right"].set_linewidth(1.0)
    ax.tick_params(width=1.1)


def jitter(x, rng):
    return x + rng.normal(0.0, JITTER_X, size=len(x))


def reduction_panel(ax, observed, synth, tau, panel_label, rng):
    regs = [r for r, _ in MODEL_RUNGS]
    labels = [l for _, l in MODEL_RUNGS]
    x = np.arange(len(regs))

    obs_vals = np.array([
        observed.loc[(observed.regressors == r) & (observed.tau == tau),
                     "pct_reduction"].iloc[0]
        for r in regs
    ])
    ax.plot(x, obs_vals, color=COLOR_OBS_LINE, linewidth=2.2, zorder=9)
    ax.scatter(x, obs_vals, s=22,
               facecolors=OBS_FACE, edgecolors=OBS_EDGE, linewidths=0.7,
               marker="o", zorder=10)

    for gen, color in [("embedded", COLOR_EMBEDDED), ("features", COLOR_FEATURES)]:
        for run in sorted(synth.run.unique()):
            sub = synth[(synth.generator == gen) & (synth.run == run) & (synth.tau == tau)]
            pts = np.array([sub.loc[sub.regressors == r, "pct_reduction"].iloc[0] for r in regs])
            ax.scatter(jitter(x, rng), pts, s=17, color=color, alpha=SIM_ALPHA, zorder=5)

    ax.set_title(rf"({panel_label}) Reduction in Criterion Function ($\tau={tau}$)",
                 fontweight="bold")
    ax.set_ylabel("Reduction (%)")
    ax.set_xlabel(r"Model Complexity $\rightarrow$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle=":", alpha=0.6)
    style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--out", default="figure_D1")
    args = ap.parse_args()

    base = Path(args.results_dir)
    observed = pd.read_csv(base / "observed_table5.csv")
    synth    = pd.read_csv(base / "synthetic_raw_C_tau.csv")

    rng = np.random.default_rng(SEED)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, constrained_layout=True
    )
    reduction_panel(ax1, observed, synth, 0.5,  "a", rng)
    reduction_panel(ax2, observed, synth, 0.75, "b", rng)

    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
