"""Matplotlib helpers for the diagnostic plots produced by ``main.py``.

Kept deliberately simple (one figure per function, saved as PNG) since the
point of this project is the PCA / trading-strategy methodology, not the
plotting layer.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless / non-interactive runs
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, out_path: Path, dpi: int = 150):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_series(series: pd.Series, title: str, out_path: Path, ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(12, 6))
    series.plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_ar_delta(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    df[["AR_1yr", "AR_15d"]].plot(ax=axes[0], linewidth=2)
    axes[0].set_title("Absorption Ratio: 1-year vs. 15-day moving average")
    axes[0].grid(alpha=0.3)

    df["AR_delta"].plot(ax=axes[1], linewidth=2, color="darkred")
    axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
    axes[1].axhline(-1.0, linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("AR Delta (standardized shift in Absorption Ratio)")
    axes[1].grid(alpha=0.3)
    _save(fig, out_path)


def plot_weights(wgts: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(wgts.index, wgts["EQ"], wgts["FI"], labels=["EQ", "FI"], alpha=0.8)
    ax.set_title("AR-Delta Strategy Allocation Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    _save(fig, out_path)


def plot_cumulative_growth(curves: dict, out_path: Path, title: str = "Cumulative Growth of $1"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in curves.items():
        series.plot(ax=ax, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_eigen_portfolio_variance(explained_variance_ratio: pd.Series, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    explained_variance_ratio.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Variance Explained by Each Eigen-Portfolio")
    ax.set_ylabel("Explained Variance Ratio")
    _save(fig, out_path)
