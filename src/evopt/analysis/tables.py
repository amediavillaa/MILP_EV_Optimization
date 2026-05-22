# src/evopt/analysis/tables.py
"""Summary tables for benchmark results — each function saves .csv and .tex."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evopt.analysis.utils import ci95

_METRICS = [
    "net_profit", "total_revenue", "total_cost",
    "served_customers", "rejected_customers", "mean_soc_fulfillment",
    "gap_to_best", "total_compute_s", "mean_step_ms",
]

_ROUND = {
    "net_profit": 2, "total_revenue": 2, "total_cost": 2,
    "served_customers": 1, "rejected_customers": 1,
    "mean_soc_fulfillment": 3, "gap_to_best": 2,
    "total_compute_s": 2, "mean_step_ms": 2,
    "profit_per_compute_s": 3,
}


def _save(df: pd.DataFrame, stem: str, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv")
    try:
        df.to_latex(output_dir / f"{stem}.tex", float_format="%.3f")
    except Exception:
        pass


def _agg_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate metric columns with mean/std/min/max/median/n/CI."""
    metrics = [c for c in _METRICS if c in df.columns]
    records = []
    for keys, grp in df.groupby(group_cols):
        row: dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for m in metrics:
            s = grp[m].dropna()
            if len(s) == 0:
                continue
            lo, hi = ci95(s)
            row[f"{m}_mean"]     = round(s.mean(), _ROUND.get(m, 3))
            row[f"{m}_std"]      = round(s.std(ddof=1), _ROUND.get(m, 3))
            row[f"{m}_min"]      = round(s.min(), _ROUND.get(m, 3))
            row[f"{m}_max"]      = round(s.max(), _ROUND.get(m, 3))
            row[f"{m}_median"]   = round(s.median(), _ROUND.get(m, 3))
            row[f"{m}_n"]        = int(len(s))
            row[f"{m}_ci_lower"] = round(lo, _ROUND.get(m, 3))
            row[f"{m}_ci_upper"] = round(hi, _ROUND.get(m, 3))
        records.append(row)
    return pd.DataFrame(records).set_index(group_cols)


def make_full_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Mean ± CI for all metrics grouped by (controller, ports)."""
    result = _agg_metrics(df, ["controller", "ports"])
    _save(result, "table_full", output_dir)
    return result


def make_cross_port_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Pivot: controllers × ports showing mean net_profit/served/rejected."""
    metrics = [m for m in ["net_profit", "served_customers", "rejected_customers"]
               if m in df.columns]
    pivot = df.groupby(["controller", "ports"])[metrics].mean().unstack("ports")
    pivot = pivot.round(2)
    _save(pivot, "table_cross_port", output_dir)
    return pivot


def make_best_controller_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Best controller per port by mean net_profit."""
    mean_profit = df.groupby(["controller", "ports"])["net_profit"].mean()
    best = mean_profit.groupby("ports").idxmax().apply(lambda x: x[0])
    best_vals = mean_profit.groupby("ports").max().round(2)
    result = pd.DataFrame({"best_controller": best, "mean_net_profit": best_vals})
    _save(result, "table_best", output_dir)
    return result


def make_compute_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """total_compute_s and mean_step_ms by controller."""
    cols = [c for c in ["total_compute_s", "mean_step_ms"] if c in df.columns]
    result = df.groupby("controller")[cols].mean().round(3)
    _save(result, "table_compute", output_dir)
    return result


def make_efficiency_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """profit_per_compute_s with CI by (controller, ports)."""
    if "profit_per_compute_s" not in df.columns:
        return pd.DataFrame()
    result = _agg_metrics(df[["controller", "ports", "profit_per_compute_s"]
                              ].copy(), ["controller", "ports"])
    _save(result, "table_efficiency", output_dir)
    return result


def make_tradeoff_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """net_profit, mean_step_ms, served_customers, rejected_customers side by side."""
    cols = [c for c in ["net_profit", "mean_step_ms",
                         "served_customers", "rejected_customers"] if c in df.columns]
    result = df.groupby(["controller", "ports"])[cols].mean().round(2)
    _save(result, "table_tradeoff", output_dir)
    return result


def make_robustness_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """CV, IQR, worst/best case net_profit per (controller, ports)."""
    if "net_profit" not in df.columns:
        return pd.DataFrame()
    rows = []
    for (ctrl, ports), grp in df.groupby(["controller", "ports"]):
        s = grp["net_profit"].dropna()
        std = s.std(ddof=1)
        mean = s.mean()
        rows.append({
            "controller": ctrl,
            "ports": ports,
            "cv":         round(std / mean if mean != 0 else np.nan, 4),
            "iqr":        round(s.quantile(0.75) - s.quantile(0.25), 3),
            "worst_case": round(s.min(), 2),
            "best_case":  round(s.max(), 2),
        })
    result = pd.DataFrame(rows).set_index(["controller", "ports"])
    _save(result, "table_robustness", output_dir)
    return result
