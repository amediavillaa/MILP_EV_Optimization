# src/milp_ev_opt/analysis/loader.py
"""Load and clean benchmark result CSVs/JSONs into a tidy per-seed DataFrame."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from milp_ev_opt.analysis.utils import extract_horizon

_METRIC_COLS = [
    "net_profit", "total_revenue", "total_cost",
    "served_customers", "rejected_customers", "mean_soc_fulfillment",
    "gap_to_best", "total_compute_s", "mean_step_ms",
]


def load_results(path: Path) -> pd.DataFrame:
    """Load a benchmark CSV or JSON into a DataFrame.

    Raises ValueError for unrecognised extensions.
    Supports multi-experiment files (multiple experiment_id values).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path, orient="records")
    raise ValueError(f"Unsupported file format: '{suffix}'. Use .csv or .json.")


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of *df*.

    1. Drops rows where all metric columns are NaN.
    2. Coerces metric columns to float.
    3. Recomputes gap_to_best within (experiment_id, seed, ports) groups so
       the gap is always ≤ 0 and is not contaminated by other experiments or
       port configurations.
    4. Warns if the DataFrame spans multiple experiment_ids.
    """
    df = df.copy()

    # Coerce metrics
    for col in _METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no useful data
    present = [c for c in _METRIC_COLS if c in df.columns]
    df = df.dropna(subset=present, how="all").reset_index(drop=True)

    # Warn on multi-experiment data
    if "experiment_id" in df.columns:
        unique_exp = df["experiment_id"].nunique()
        if unique_exp > 1:
            warnings.warn(
                f"DataFrame contains {unique_exp} distinct experiment_id values. "
                "Tables and charts aggregate across all of them. "
                "Filter by experiment_id before drawing per-experiment conclusions.",
                UserWarning,
                stacklevel=2,
            )

    # Recompute gap_to_best within experimental conditions.
    # storage.build_summary_from_results() uses a global best which silently
    # mixes port configurations and experiments — we correct that here.
    group_cols = (
        ["experiment_id", "seed", "ports"]
        if "experiment_id" in df.columns
        else ["seed", "ports"]
    )
    if "net_profit" in df.columns and all(c in df.columns for c in group_cols):
        df["gap_to_best"] = (
            df["net_profit"]
            - df.groupby(group_cols)["net_profit"].transform("max")
        )

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add horizon (int from milp_hN) and profit_per_compute_s (NaN-safe)."""
    df = df.copy()
    df["horizon"] = df["controller"].map(extract_horizon)
    # Zero-compute baselines get NaN, not inf
    compute = df["total_compute_s"].replace(0.0, np.nan)
    df["profit_per_compute_s"] = df["net_profit"] / compute
    return df
