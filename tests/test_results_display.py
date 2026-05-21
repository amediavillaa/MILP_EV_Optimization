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


# ── utils ──────────────────────────────────────────────────────────────────
from evopt.analysis.utils import (
    extract_horizon, format_experiment_id, ci95, colorblind_palette,
)

def test_extract_horizon_milp():
    assert extract_horizon("milp_h6") == 6

def test_extract_horizon_double_digit():
    assert extract_horizon("milp_h12") == 12

def test_extract_horizon_baseline():
    assert math.isnan(extract_horizon("equal_share"))

def test_extract_horizon_baseline_random():
    assert math.isnan(extract_horizon("random"))

def test_ci95_known_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = ci95(s)
    mean, std, n = s.mean(), s.std(ddof=1), len(s)
    expected_lo = mean - 1.96 * std / math.sqrt(n)
    expected_hi = mean + 1.96 * std / math.sqrt(n)
    assert abs(lo - expected_lo) < 1e-9
    assert abs(hi - expected_hi) < 1e-9

def test_ci95_single_value():
    s = pd.Series([7.0, 7.0, 7.0])
    lo, hi = ci95(s)
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)

def test_colorblind_palette_length():
    assert len(colorblind_palette(4)) == 4

def test_colorblind_palette_cycles():
    assert len(colorblind_palette(10)) == 10

def test_format_experiment_id_basic():
    eid = format_experiment_id(ports=[3, 6], horizons=[1, 6], tariff="dynamic:1.3")
    assert "dynamic_1_3" in eid or "dynamic" in eid
    assert "p3_6" in eid
    assert "h1_6" in eid

from evopt.analysis.utils import save_metadata

def test_save_metadata_contains_keys(tmp_path):
    cfg = {
        "experiment_id": "test", "ports": [3], "horizons": [1],
        "n_seeds": 5, "tariff": "0.75", "bess_enabled": True,
        "v2g_enabled": False, "solver": "highs", "cli_command": "pytest",
    }
    save_metadata(cfg, tmp_path)
    import json
    data = json.loads((tmp_path / "metadata.json").read_text())
    for key in ["experiment_id", "timestamp", "git_commit", "hostname",
                "python_version", "ports", "horizons", "n_seeds",
                "tariff", "bess_enabled", "v2g_enabled", "solver"]:
        assert key in data, f"missing key: {key}"
