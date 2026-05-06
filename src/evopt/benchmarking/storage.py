from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evopt.benchmarking.results import ChargaxSimResults, StepRecord


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


def build_summary(results_dir: Path) -> pd.DataFrame:
    rows = []
    for json_file in sorted(Path(results_dir).glob("**/*.json")):
        result = load(json_file)
        rows.append({
            "controller":           result.controller_name,
            "seed":                 result.seed,
            "net_profit":           result.net_profit,
            "total_revenue":        result.total_revenue,
            "total_cost":           result.total_cost,
            "served_customers":     result.served_customers,
            "rejected_customers":   result.rejected_customers,
            "mean_soc_at_departure": result.mean_soc_at_departure,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    best = df.groupby("controller")["net_profit"].mean().max()
    df["gap_to_best"] = df["net_profit"] - best
    return df
