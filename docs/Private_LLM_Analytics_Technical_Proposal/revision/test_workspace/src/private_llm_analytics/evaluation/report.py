"""Plotting and report generation routines for research experiments."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd


def set_matplotlib_style():
    """Configure clean publication-style matplotlib parameters."""
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.color": "#dddddd",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def plot_single_line_chart(
    x: List[float],
    y_series: Dict[str, List[float]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Union[str, Path],
    x_log: bool = False,
    y_log: bool = False,
) -> None:
    """Plot a single clean publication-ready figure and save to disk."""
    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    markers = ["o", "s", "^", "D", "v", "p"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (label, y_vals) in enumerate(y_series.items()):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        ax.plot(x, y_vals, label=label, marker=marker, color=color, alpha=0.9)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # Save PNG
    plt.savefig(out_p.with_suffix(".png"), dpi=300)
    # Also save PDF
    plt.savefig(out_p.with_suffix(".pdf"))
    plt.close(fig)


def save_summary_table(
    summary_data: Union[pd.DataFrame, List[Dict]],
    output_path: Union[str, Path],
) -> pd.DataFrame:
    """Save experiment summary table to CSV."""
    if isinstance(summary_data, pd.DataFrame):
        df = summary_data
    else:
        df = pd.DataFrame(summary_data)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    return df
