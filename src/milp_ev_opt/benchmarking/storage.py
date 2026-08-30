from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from milp_ev_opt.benchmarking.results import ChargaxSimResults, StepRecord


def save(result: ChargaxSimResults, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)


def load(path: Path) -> ChargaxSimResults:
    with open(path) as f:
        data = json.load(f)
    step_log = [StepRecord(**s) for s in data.pop("step_log")]
    return ChargaxSimResults(**data, step_log=step_log)


def build_summary_from_results(results: list[ChargaxSimResults]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows = [
        {
            "controller":            r.controller_name,
            "seed":                  r.seed,
            "net_profit":            r.net_profit,
            "total_revenue":         r.total_revenue,
            "total_cost":            r.total_cost,
            "served_customers":      r.served_customers,
            "rejected_customers":    r.rejected_customers,
            "mean_soc_fulfillment":  r.mean_soc_fulfillment,
            "total_compute_s":       r.total_compute_s,
            "mean_step_ms":          r.mean_step_ms,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    best = df.groupby("controller")["net_profit"].mean().max()
    df["gap_to_best"] = df["net_profit"] - best
    return df


def build_summary(results_dir: Path) -> pd.DataFrame:
    results = [load(p) for p in sorted(Path(results_dir).glob("**/*.json"))]
    return build_summary_from_results(results)
