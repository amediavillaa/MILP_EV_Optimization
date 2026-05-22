# src/evopt/analysis/correlation.py
"""Pearson correlation matrix and summary for benchmark metrics."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_EXCLUDE = {"seed", "ports", "horizon", "experiment_id",
            "controller", "tariff", "solver"}


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix on numeric metric columns.

    Excludes non-metric columns (seed, ports, horizon, controller, etc.)
    so the matrix reflects relationships between performance metrics only.
    """
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _EXCLUDE
    ]
    return df[num_cols].corr(method="pearson")


def save_correlation_csv(corr: pd.DataFrame, output_dir: Path) -> None:
    """Write the Pearson matrix to correlation_matrix.csv."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    corr.round(4).to_csv(Path(output_dir) / "correlation_matrix.csv")


def summarise_correlations(corr: pd.DataFrame) -> str:
    """Return a Markdown string highlighting the top 5 positive/negative pairs."""
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((corr.loc[a, b], a, b))
    pairs.sort(key=lambda x: x[0])

    neg5 = pairs[:5]
    pos5 = pairs[-5:][::-1]

    lines = ["## Correlation Summary\n",
             "**Top 5 positive correlations:**\n"]
    for r, a, b in pos5:
        lines.append(f"- `{a}` ↔ `{b}`: r = {r:.3f}\n")
    lines.append("\n**Top 5 negative correlations:**\n")
    for r, a, b in neg5:
        lines.append(f"- `{a}` ↔ `{b}`: r = {r:.3f}\n")
    return "".join(lines)
