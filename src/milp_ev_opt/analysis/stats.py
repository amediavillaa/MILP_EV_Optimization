# src/milp_ev_opt/analysis/stats.py
"""Paired statistical tests for benchmark controller comparisons.

All tests (paired t-test, Wilcoxon signed-rank, paired Cohen's d) operate on
matched observations aligned by seed and ports (and experiment_id if present).
Using pooled-variance Cohen's d would be incorrect here because observations are
paired by experimental condition — unmatched rows are dropped with a warning.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _resolve_match_on(df: pd.DataFrame) -> list[str]:
    """Return match_on columns, including experiment_id if the column exists."""
    base = ["seed", "ports"]
    if "experiment_id" in df.columns:
        return ["experiment_id"] + base
    return base


def compare_controllers(
    df: pd.DataFrame,
    metric: str,
    controller_a: str,
    controller_b: str,
    match_on: list[str] | None = None,
) -> dict:
    """Paired t-test, Wilcoxon, and paired Cohen's d for metric between two controllers.

    Observations are aligned by *match_on* (inner join). Unmatched rows are
    dropped; a warning is emitted when this reduces the pair count, because all
    three statistics require the same number of paired observations.

    cohen_d = mean(diff) / std(diff)  — paired formulation, not pooled-variance.
    """
    if match_on is None:
        match_on = _resolve_match_on(df)

    a_df = df[df["controller"] == controller_a][match_on + [metric]].copy()
    b_df = df[df["controller"] == controller_b][match_on + [metric]].copy()

    merged = a_df.merge(b_df, on=match_on, suffixes=("_a", "_b"))
    n_pairs = len(merged)

    total_a = len(a_df)
    if n_pairs < total_a:
        warnings.warn(
            f"compare_controllers: {total_a - n_pairs} rows from '{controller_a}' "
            f"had no match in '{controller_b}' on {match_on} and were dropped.",
            UserWarning,
            stacklevel=2,
        )

    col_a = f"{metric}_a"
    col_b = f"{metric}_b"

    if n_pairs == 0:
        return {
            "n_pairs": 0, "mean_a": np.nan, "mean_b": np.nan,
            "mean_diff": np.nan, "p_ttest": np.nan, "p_wilcoxon": np.nan,
            "cohen_d": np.nan, "is_significant": False,
        }

    a_vals = merged[col_a].values
    b_vals = merged[col_b].values
    diffs = a_vals - b_vals

    if controller_a == controller_b or np.all(diffs == 0):
        return {
            "n_pairs": n_pairs,
            "mean_a": float(a_vals.mean()),
            "mean_b": float(b_vals.mean()),
            "mean_diff": 0.0,
            "p_ttest": 1.0,
            "p_wilcoxon": 1.0,
            "cohen_d": 0.0,
            "is_significant": False,
        }

    _, p_t = scipy_stats.ttest_rel(a_vals, b_vals)
    try:
        _, p_w = scipy_stats.wilcoxon(diffs)
    except ValueError:
        p_w = np.nan

    std_diff = diffs.std(ddof=1)
    cohen_d = float(diffs.mean() / std_diff) if std_diff > 0 else 0.0

    return {
        "n_pairs":        n_pairs,
        "mean_a":         float(a_vals.mean()),
        "mean_b":         float(b_vals.mean()),
        "mean_diff":      float(diffs.mean()),
        "p_ttest":        float(p_t),
        "p_wilcoxon":     float(p_w),
        "cohen_d":        cohen_d,
        "is_significant": bool(p_t < 0.05),
    }


def compare_all_controllers(
    df: pd.DataFrame,
    metric: str,
    match_on: list[str] | None = None,
) -> pd.DataFrame:
    """Pairwise compare_controllers for every unique controller pair.

    Returns a long-form DataFrame with one row per pair.
    match_on is resolved the same way as compare_controllers.
    """
    if match_on is None:
        match_on = _resolve_match_on(df)

    controllers = sorted(df["controller"].unique())
    records = []
    for a, b in combinations(controllers, 2):
        row = compare_controllers(df, metric, a, b, match_on=match_on)
        row["controller_a"] = a
        row["controller_b"] = b
        row["metric"] = metric
        records.append(row)

    cols = ["controller_a", "controller_b", "metric",
            "n_pairs", "mean_a", "mean_b", "mean_diff",
            "p_ttest", "p_wilcoxon", "cohen_d", "is_significant"]
    return pd.DataFrame(records)[cols]


def save_significance_csv(results: pd.DataFrame, output_dir: Path) -> None:
    """Write table_significance.csv and table_significance.tex."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "table_significance.csv", index=False)
    try:
        results.to_latex(output_dir / "table_significance.tex",
                         index=False, float_format="%.4f")
    except Exception:
        pass


def save_significance_summary(results: pd.DataFrame, output_dir: Path) -> None:
    """Write a human-readable significance_summary.md."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sig = results[results["is_significant"]]
    lines = ["# Statistical Significance Summary\n\n"]
    lines.append(f"Metric: `{results['metric'].iloc[0]}`\n\n")
    lines.append(f"Total pairs tested: {len(results)}  \n")
    lines.append(f"Significant pairs (p < 0.05): {len(sig)}\n\n")

    if not sig.empty:
        lines.append("## Significant Differences\n\n")
        for _, row in sig.iterrows():
            lines.append(
                f"- **{row['controller_a']}** vs **{row['controller_b']}**: "
                f"mean diff = {row['mean_diff']:.3f}, "
                f"p (t-test) = {row['p_ttest']:.4f}, "
                f"Cohen's d = {row['cohen_d']:.3f}\n"
            )
    else:
        lines.append("No statistically significant differences found.\n")

    (output_dir / "significance_summary.md").write_text("".join(lines))
