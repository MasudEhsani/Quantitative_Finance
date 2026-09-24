"""Matplotlib helpers for the IRL/GMR project's diagnostic plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, out_path: Path, dpi: int = 150):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_level_with_smas(X: pd.DataFrame, short_roll: pd.DataFrame, long_roll: pd.DataFrame,
                          ticker: str, start_date: str, end_date: str, out_path: Path,
                          window_short: int, window_long: int, ylabel: str = "Cap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X.loc[start_date:end_date].index, X.loc[start_date:end_date, ticker], label=ticker)
    ax.plot(long_roll.loc[start_date:end_date].index, long_roll.loc[start_date:end_date, ticker],
            label=f"{window_long}-day SMA")
    ax.plot(short_roll.loc[start_date:end_date].index, short_roll.loc[start_date:end_date, ticker],
            label=f"{window_short}-day SMA")
    ax.legend(loc="best")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ticker}: level and moving averages")
    _save(fig, out_path)


def plot_residual_hist(residuals: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals.values.ravel(), bins=60, density=True, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("residual")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_kappa_by_asset(kappa: pd.Series, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    kappa.sort_values().plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_ylabel("kappa")
    _save(fig, out_path)


def plot_sigma_heatmap(Sigma: pd.DataFrame, title: str, out_path: Path, max_assets: int = 40):
    S = Sigma.iloc[:max_assets, :max_assets]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(S.values, cmap="RdBu_r", vmin=-np.abs(S.values).max(), vmax=np.abs(S.values).max())
    ax.set_xticks(range(len(S.columns)))
    ax.set_xticklabels(S.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(S.index)))
    ax.set_yticklabels(S.index, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out_path)


def plot_model_comparison(table: pd.DataFrame, out_path: Path, metric: str = "BIC"):
    fig, ax = plt.subplots(figsize=(9, 5))
    table[metric].sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Model comparison ({metric}, lower is better)")
    ax.set_xlabel(metric)
    _save(fig, out_path)


def plot_cumulative_growth(curves: dict, out_path: Path, title: str = "Cumulative Growth", log_scale: bool = True):
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in curves.items():
        series.plot(ax=ax, linewidth=2, label=label)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log scale)")
    _save(fig, out_path)
