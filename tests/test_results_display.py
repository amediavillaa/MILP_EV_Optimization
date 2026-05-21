# tests/test_results_display.py
from __future__ import annotations
import math, warnings
import numpy as np
import pandas as pd
import pytest

def _sample_df(n_seeds: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ctrl, base in [("milp_h1", 8.0), ("milp_h6", 12.0), ("equal_share", 5.0)]:
        for ports in [3, 6]:
            for seed in range(n_seeds):
                rows.append({
                    "experiment_id":       "exp_a",
                    "controller":          ctrl,
                    "seed":                seed,
                    "ports":               ports,
                    "tariff":              "0.75",
                    "bess_enabled":        True,
                    "v2g_enabled":         False,
                    "solver":              "highs",
                    "net_profit":          rng.normal(base, 1.5),
                    "total_revenue":       rng.normal(base + 10, 2.0),
                    "total_cost":          rng.normal(10.0, 0.5),
                    "served_customers":    float(rng.integers(3, 9)),
                    "rejected_customers":  float(rng.integers(0, 3)),
                    "mean_soc_fulfillment": rng.uniform(0.7, 1.0),
                    "gap_to_best":         0.0,
                    "total_compute_s":     0.0 if ctrl == "equal_share" else rng.uniform(0.1, 2.0),
                    "mean_step_ms":        0.0 if ctrl == "equal_share" else rng.uniform(1.0, 50.0),
                })
    return pd.DataFrame(rows)
