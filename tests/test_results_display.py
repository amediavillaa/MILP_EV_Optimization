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


# ── loader ─────────────────────────────────────────────────────────────────
from evopt.analysis.loader import load_results, clean_results, add_derived_metrics

def test_load_results_csv(tmp_path):
    df = _sample_df()
    p = tmp_path / "bench.csv"
    df.to_csv(p, index=False)
    loaded = load_results(p)
    assert len(loaded) == len(df)
    assert "controller" in loaded.columns

def test_load_results_json(tmp_path):
    df = _sample_df()
    p = tmp_path / "bench.json"
    df.to_json(p, orient="records", indent=2)
    loaded = load_results(p)
    assert len(loaded) == len(df)

def test_load_results_unknown_format(tmp_path):
    p = tmp_path / "bench.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_results(p)

def test_gap_to_best_always_nonpositive():
    df = clean_results(_sample_df())
    assert (df["gap_to_best"] <= 1e-9).all()

def test_gap_to_best_best_controller_is_zero():
    df = clean_results(_sample_df())
    maxes = df.groupby(["experiment_id", "seed", "ports"])["gap_to_best"].max()
    assert (maxes.abs() < 1e-9).all()

def test_gap_to_best_multi_experiment():
    df1 = _sample_df(); df1["experiment_id"] = "exp_a"
    df2 = _sample_df(); df2["experiment_id"] = "exp_b"
    df2["net_profit"] += 1000.0
    combined = pd.concat([df1, df2], ignore_index=True)
    cleaned = clean_results(combined)
    exp_a = cleaned[cleaned["experiment_id"] == "exp_a"]
    maxes = exp_a.groupby(["seed", "ports"])["gap_to_best"].max()
    assert (maxes.abs() < 1e-9).all()

def test_clean_results_warns_multi_experiment():
    df1 = _sample_df(); df1["experiment_id"] = "exp_a"
    df2 = _sample_df(); df2["experiment_id"] = "exp_b"
    combined = pd.concat([df1, df2], ignore_index=True)
    with pytest.warns(UserWarning, match="multiple experiment"):
        clean_results(combined)

def test_derived_metrics_columns_exist():
    df = add_derived_metrics(_sample_df())
    assert "horizon" in df.columns
    assert "profit_per_compute_s" in df.columns

def test_extract_horizon_in_derived():
    df = add_derived_metrics(_sample_df())
    h1_rows = df[df["controller"] == "milp_h1"]
    assert (h1_rows["horizon"] == 1).all()
    baseline_rows = df[df["controller"] == "equal_share"]
    assert baseline_rows["horizon"].isna().all()

def test_profit_per_compute_zero():
    df = add_derived_metrics(_sample_df())
    zero_rows = df[df["total_compute_s"] == 0.0]
    assert zero_rows["profit_per_compute_s"].isna().all()

def test_profit_per_compute_not_inf():
    df = add_derived_metrics(_sample_df())
    assert not np.isinf(df["profit_per_compute_s"].fillna(0)).any()
