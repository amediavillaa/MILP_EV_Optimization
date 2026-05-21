# Results Display Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evopt.analysis`, a standalone research-analysis package that reads benchmark CSVs and produces publication-quality tables, charts, statistical tests, and Markdown/HTML reports for an EV charging thesis.

**Architecture:** Two-phase pipeline — `run_chargax_benchmark.py` gains `--save PATH` to export per-seed CSVs with config metadata; `evopt.analysis` reads that CSV and orchestrates loader → tables → plots → correlation → stats → report. All analysis modules import only from `utils.py`; no cross-module imports within the package.

**Tech Stack:** Python 3.11, pandas ≥ 2.0, matplotlib ≥ 3.7, seaborn ≥ 0.12, scipy ≥ 1.10, pathlib, argparse, re, json, base64.

---

## Shared test fixture (used in every task)

Every task that writes tests uses the helper below. Add it once at the top of `tests/test_results_display.py` in Task 1 and **do not repeat it** in later tasks.

```python
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
                    "gap_to_best":         0.0,          # recomputed by clean_results
                    "total_compute_s":     0.0 if ctrl == "equal_share" else rng.uniform(0.1, 2.0),
                    "mean_step_ms":        0.0 if ctrl == "equal_share" else rng.uniform(1.0, 50.0),
                })
    return pd.DataFrame(rows)
```

---

## Task 1: Package scaffold and dependencies

**Files:**
- Create: `src/evopt/analysis/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_results_display.py` (fixture only for now)

- [ ] **Step 1: Add dependencies to pyproject.toml**

Open `pyproject.toml`. In the `dependencies` list, add three entries:

```toml
[project]
dependencies = [
    "pyomo>=6.0",
    "pandas>=2.0",
    "jax>=0.4.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scipy>=1.10",
]
```

- [ ] **Step 2: Create the analysis package**

```python
# src/evopt/analysis/__init__.py
"""EV benchmark results analysis pipeline."""
```

- [ ] **Step 3: Create test file with shared fixture**

Create `tests/test_results_display.py` with the shared fixture shown in the header above. No test functions yet.

- [ ] **Step 4: Verify import works**

```
python -c "import evopt.analysis"
```

Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/__init__.py pyproject.toml tests/test_results_display.py
git commit -m "feat: scaffold evopt.analysis package and add matplotlib/seaborn/scipy deps"
```

---

## Task 2: utils.py

**Files:**
- Create: `src/evopt/analysis/utils.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
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
    # std=0 → CI collapses to the mean
    s = pd.Series([7.0, 7.0, 7.0])
    lo, hi = ci95(s)
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)

def test_colorblind_palette_length():
    assert len(colorblind_palette(4)) == 4

def test_colorblind_palette_cycles():
    # requesting more than 8 should not raise
    assert len(colorblind_palette(10)) == 10

def test_format_experiment_id_basic():
    eid = format_experiment_id(ports=[3, 6], horizons=[1, 6], tariff="dynamic:1.3")
    assert "dynamic_1_3" in eid or "dynamic" in eid
    assert "p3_6" in eid
    assert "h1_6" in eid
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "utils or extract or ci95 or colorblind or format_exp" -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement utils.py**

```python
# src/evopt/analysis/utils.py
"""Shared utilities: styling, I/O helpers, statistical helpers, ID generation."""
from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Okabe-Ito colorblind-safe palette (8 colours)
_OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


def apply_pub_style() -> None:
    """Set publication-quality rcParams: font 12, clean spines, tight layout."""
    matplotlib.rcParams.update({
        "font.size":        12,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "figure.figsize":   (8, 5),
        "figure.dpi":       100,
        "savefig.dpi":      300,
        "figure.autolayout": True,
    })


def save_figure(fig: Figure, path: Path, dpi: int = 300) -> None:
    """Save *fig* to *path* at *dpi*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def extract_horizon(controller_name: str) -> float:
    """Return horizon int from 'milp_hN', or NaN for non-MILP controllers."""
    m = re.fullmatch(r"milp_h(\d+)", controller_name)
    return float(m.group(1)) if m else float("nan")


def format_experiment_id(
    ports: list[int],
    horizons: list[int],
    tariff: str,
    **_kwargs,
) -> str:
    """Build a human-readable experiment ID, e.g. 'dynamic_1_3_p3_6_h1_6'."""
    safe_tariff = re.sub(r"[^a-zA-Z0-9]", "_", tariff)
    p_str = "_".join(str(p) for p in sorted(ports))
    h_str = "_".join(str(h) for h in sorted(horizons))
    return f"{safe_tariff}_p{p_str}_h{h_str}"


def ci95(series: pd.Series) -> tuple[float, float]:
    """Return (ci_lower, ci_upper) 95% confidence interval for *series*."""
    n = len(series)
    mean = series.mean()
    if n < 2:
        return float(mean), float(mean)
    std = series.std(ddof=1)
    margin = 1.96 * std / np.sqrt(n)
    return float(mean - margin), float(mean + margin)


def colorblind_palette(n: int) -> list[str]:
    """Return *n* colours from the Okabe-Ito palette, cycling if n > 8."""
    return [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(n)]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_metadata(config: dict, output_dir: Path) -> None:
    """Write metadata.json to *output_dir* with reproducibility fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_id":  config.get("experiment_id", ""),
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "git_commit":     _git_hash(),
        "cli_command":    config.get("cli_command", ""),
        "hostname":       socket.gethostname(),
        "python_version": sys.version.split()[0],
        "ports":          config.get("ports", []),
        "horizons":       config.get("horizons", []),
        "n_seeds":        config.get("n_seeds", 0),
        "tariff":         config.get("tariff", ""),
        "bess_enabled":   config.get("bess_enabled", False),
        "v2g_enabled":    config.get("v2g_enabled", False),
        "solver":         config.get("solver", ""),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "utils or extract or ci95 or colorblind or format_exp" -v
```

Expected: all green.

- [ ] **Step 5: Add metadata test, run**

Append to `tests/test_results_display.py`:

```python
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
```

```
python -m pytest tests/test_results_display.py::test_save_metadata_contains_keys -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```
git add src/evopt/analysis/utils.py tests/test_results_display.py
git commit -m "feat: add evopt.analysis.utils (styling, CI, horizon extraction, metadata)"
```

---

## Task 3: loader.py

**Files:**
- Create: `src/evopt/analysis/loader.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
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
    # Within each (experiment_id, seed, ports) group the max gap_to_best == 0
    maxes = df.groupby(["experiment_id", "seed", "ports"])["gap_to_best"].max()
    assert (maxes.abs() < 1e-9).all()

def test_gap_to_best_multi_experiment():
    df1 = _sample_df(); df1["experiment_id"] = "exp_a"
    df2 = _sample_df(); df2["experiment_id"] = "exp_b"
    # In exp_b, inflate all profits so they would dominate exp_a if mixed
    df2["net_profit"] += 1000.0
    combined = pd.concat([df1, df2], ignore_index=True)
    cleaned = clean_results(combined)
    # exp_a rows must still have a best with gap_to_best == 0, not inflated by exp_b
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
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "loader or load_results or gap_to_best or derived or profit_per_compute" -v 2>&1 | head -20
```

Expected: `ImportError`.

- [ ] **Step 3: Implement loader.py**

```python
# src/evopt/analysis/loader.py
"""Load and clean benchmark result CSVs/JSONs into a tidy per-seed DataFrame."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evopt.analysis.utils import extract_horizon

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
                f"DataFrame contains {unique_exp} multiple experiment_id values. "
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
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "loader or load_results or gap_to_best or derived or profit_per_compute or multi_exp or warns" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/loader.py tests/test_results_display.py
git commit -m "feat: add evopt.analysis.loader with gap_to_best correction and derived metrics"
```

---

## Task 4: tables.py

**Files:**
- Create: `src/evopt/analysis/tables.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
# ── tables ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
from evopt.analysis.tables import (
    make_full_table, make_cross_port_table, make_best_controller_table,
    make_compute_table, make_efficiency_table, make_tradeoff_table,
    make_robustness_table,
)
from evopt.analysis.loader import clean_results, add_derived_metrics

def _ready_df():
    return add_derived_metrics(clean_results(_sample_df()))

def test_make_full_table_produces_csv(tmp_path):
    result = make_full_table(_ready_df(), tmp_path)
    assert (tmp_path / "table_full.csv").exists()
    assert len(result) > 0

def test_make_full_table_produces_tex(tmp_path):
    make_full_table(_ready_df(), tmp_path)
    tex = (tmp_path / "table_full.tex").read_text()
    assert len(tex) > 10

def test_make_full_table_has_stat_columns(tmp_path):
    result = make_full_table(_ready_df(), tmp_path)
    cols = result.columns.tolist()
    col_str = str(cols)
    for stat in ["mean", "std", "min", "max", "median", "n", "ci_lower", "ci_upper"]:
        assert stat in col_str, f"missing stat column: {stat}"

def test_make_cross_port_table(tmp_path):
    result = make_cross_port_table(_ready_df(), tmp_path)
    assert (tmp_path / "table_cross_port.csv").exists()
    assert (tmp_path / "table_cross_port.tex").exists()

def test_make_best_controller_table(tmp_path):
    result = make_best_controller_table(_ready_df(), tmp_path)
    assert (tmp_path / "table_best.csv").exists()
    # One row per port value
    assert len(result) == _ready_df()["ports"].nunique()

def test_make_robustness_table(tmp_path):
    result = make_robustness_table(_ready_df(), tmp_path)
    assert (tmp_path / "table_robustness.csv").exists()
    col_str = str(result.columns.tolist())
    assert "cv" in col_str
    assert "iqr" in col_str
    assert "worst_case" in col_str
    assert "best_case" in col_str
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "table" -v 2>&1 | head -20
```

Expected: `ImportError`.

- [ ] **Step 3: Implement tables.py**

```python
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
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "table" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/tables.py tests/test_results_display.py
git commit -m "feat: add evopt.analysis.tables with CSV+LaTeX export and CI columns"
```

---

## Task 5: plots.py — profit and performance charts

**Files:**
- Create: `src/evopt/analysis/plots.py` (partial — extended in Tasks 6, 7, 8)
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests for profit charts**

Append to `tests/test_results_display.py`:

```python
# ── plots (profit charts) ───────────────────────────────────────────────────
from evopt.analysis.plots import (
    plot_profit_by_controller,
    plot_profit_vs_horizon,
    plot_compute_vs_horizon,
)

def _plot_df():
    return add_derived_metrics(clean_results(_sample_df()))

def _png_nonempty(path):
    assert path.exists(), f"Missing: {path}"
    assert path.stat().st_size > 0, f"Empty file: {path}"

def test_plot_profit_by_controller(tmp_path):
    plot_profit_by_controller(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "profit_by_controller.png")

def test_plot_profit_vs_horizon(tmp_path):
    plot_profit_vs_horizon(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "profit_vs_horizon.png")

def test_plot_compute_vs_horizon(tmp_path):
    plot_compute_vs_horizon(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "compute_vs_horizon.png")
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "plot_profit or plot_compute_vs" -v 2>&1 | head -15
```

Expected: `ImportError`.

- [ ] **Step 3: Create plots.py with profit charts**

```python
# src/evopt/analysis/plots.py
"""Publication-quality charts for EV benchmark results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evopt.analysis.utils import apply_pub_style, colorblind_palette, save_figure


def plot_profit_by_controller(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: mean net_profit by controller, grouped by ports.

    Error bars show 95% confidence intervals over seeds.
    """
    apply_pub_style()
    ports_vals = sorted(df["ports"].unique())
    controllers = df["controller"].unique()
    palette = colorblind_palette(len(ports_vals))

    agg = (
        df.groupby(["controller", "ports"])["net_profit"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])

    x = np.arange(len(controllers))
    width = 0.8 / len(ports_vals)

    fig, ax = plt.subplots(figsize=(max(8, len(controllers) * 1.2), 5))
    for i, (port, color) in enumerate(zip(ports_vals, palette)):
        sub = agg[agg["ports"] == port].set_index("controller").reindex(controllers)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            sub["mean"].fillna(0),
            width,
            yerr=sub["ci"].fillna(0),
            label=f"{port} ports",
            color=color,
            capsize=4,
            error_kw={"elinewidth": 1},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(controllers, rotation=30, ha="right")
    ax.set_xlabel("Controller")
    ax.set_ylabel("Net Profit (€)")
    ax.set_title("Mean Net Profit by Controller (95% CI)")
    ax.legend(title="Ports")
    save_figure(fig, Path(output_dir) / "profit_by_controller.png")


def plot_profit_vs_horizon(df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart: mean net_profit vs MILP horizon, one line per ports value.

    Error bands show 95% confidence intervals over seeds.
    Only MILP controllers (milp_hN) are included.
    """
    apply_pub_style()
    milp = df[df["horizon"].notna()].copy()
    if milp.empty:
        fig, ax = plt.subplots()
        ax.set_title("No MILP data")
        save_figure(fig, Path(output_dir) / "profit_vs_horizon.png")
        return

    ports_vals = sorted(milp["ports"].unique())
    palette = colorblind_palette(len(ports_vals))

    fig, ax = plt.subplots()
    for port, color in zip(ports_vals, palette):
        sub = milp[milp["ports"] == port]
        agg = sub.groupby("horizon")["net_profit"].agg(
            mean="mean", std="std", n="count"
        ).reset_index()
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
        ax.plot(agg["horizon"], agg["mean"], marker="o", label=f"{port} ports", color=color)
        ax.fill_between(
            agg["horizon"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"],
            alpha=0.2, color=color,
        )
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Net Profit (€)")
    ax.set_title("MILP Net Profit vs Horizon (95% CI)")
    ax.legend(title="Ports")
    save_figure(fig, Path(output_dir) / "profit_vs_horizon.png")


def plot_compute_vs_horizon(df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart: mean mean_step_ms vs MILP horizon, one line per ports value.

    Error bars show 95% CI. Only MILP controllers are included.
    """
    apply_pub_style()
    milp = df[df["horizon"].notna()].copy()
    if milp.empty:
        fig, ax = plt.subplots()
        ax.set_title("No MILP data")
        save_figure(fig, Path(output_dir) / "compute_vs_horizon.png")
        return

    ports_vals = sorted(milp["ports"].unique())
    palette = colorblind_palette(len(ports_vals))

    fig, ax = plt.subplots()
    for port, color in zip(ports_vals, palette):
        sub = milp[milp["ports"] == port]
        agg = sub.groupby("horizon")["mean_step_ms"].agg(
            mean="mean", std="std", n="count"
        ).reset_index()
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
        ax.errorbar(
            agg["horizon"], agg["mean"], yerr=agg["ci"],
            marker="s", capsize=4, label=f"{port} ports", color=color,
        )
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Mean Step Time (ms)")
    ax.set_title("MILP Compute Time vs Horizon (95% CI)")
    ax.legend(title="Ports")
    save_figure(fig, Path(output_dir) / "compute_vs_horizon.png")
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "plot_profit or plot_compute_vs" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add profit and compute-vs-horizon charts to plots.py"
```

---

## Task 6: plots.py — distribution and scatter charts

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
from evopt.analysis.plots import (
    plot_profit_boxplot, plot_profit_violin, plot_compute_boxplot,
    plot_tradeoff_scatter, plot_pareto_frontier,
)

def test_plot_profit_boxplot(tmp_path):
    plot_profit_boxplot(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "profit_boxplot.png")

def test_plot_profit_violin(tmp_path):
    plot_profit_violin(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "profit_violin.png")

def test_plot_compute_boxplot(tmp_path):
    plot_compute_boxplot(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "compute_boxplot.png")

def test_plot_tradeoff_scatter(tmp_path):
    plot_tradeoff_scatter(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "tradeoff_scatter.png")

def test_plot_pareto_frontier(tmp_path):
    plot_pareto_frontier(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "pareto_frontier.png")
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "boxplot or violin or tradeoff or pareto" -v 2>&1 | head -15
```

Expected: `ImportError` (functions not yet defined).

- [ ] **Step 3: Add distribution + scatter functions to plots.py**

Append to `src/evopt/analysis/plots.py`:

```python
def plot_profit_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of per-seed net_profit per controller, coloured by ports."""
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ports_vals = sorted(df["ports"].unique())
    palette = {p: c for p, c in zip(ports_vals, colorblind_palette(len(ports_vals)))}
    import seaborn as sns
    sns.boxplot(
        data=df, x="controller", y="net_profit", hue="ports",
        palette=palette, ax=ax, flierprops={"marker": "o", "markersize": 3},
    )
    ax.set_xlabel("Controller")
    ax.set_ylabel("Net Profit (€)")
    ax.set_title("Net Profit Distribution by Controller")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Ports")
    save_figure(fig, Path(output_dir) / "profit_boxplot.png")


def plot_profit_violin(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot of per-seed net_profit per controller, coloured by ports."""
    apply_pub_style()
    import seaborn as sns
    ports_vals = sorted(df["ports"].unique())
    palette = {p: c for p, c in zip(ports_vals, colorblind_palette(len(ports_vals)))}
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df, x="controller", y="net_profit", hue="ports",
        palette=palette, ax=ax, split=False, inner="quart",
    )
    ax.set_xlabel("Controller")
    ax.set_ylabel("Net Profit (€)")
    ax.set_title("Net Profit Distribution (Violin) by Controller")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Ports")
    save_figure(fig, Path(output_dir) / "profit_violin.png")


def plot_compute_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of per-seed total_compute_s per controller."""
    apply_pub_style()
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df, x="controller", y="total_compute_s", ax=ax,
        color=colorblind_palette(1)[0],
        flierprops={"marker": "o", "markersize": 3},
    )
    ax.set_xlabel("Controller")
    ax.set_ylabel("Total Compute Time (s)")
    ax.set_title("Compute Time Distribution by Controller")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, Path(output_dir) / "compute_boxplot.png")


def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """3-panel scatter: profit/step_ms, served/rejected, soc/profit."""
    apply_pub_style()
    agg = df.groupby("controller")[
        ["net_profit", "mean_step_ms", "served_customers",
         "rejected_customers", "mean_soc_fulfillment"]
    ].mean().reset_index()

    colors = colorblind_palette(len(agg))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("mean_step_ms",        "net_profit",          "Step Time (ms)",      "Net Profit (€)",       "Profit vs Step Time"),
        ("rejected_customers",  "served_customers",    "Rejected Customers",  "Served Customers",     "Served vs Rejected"),
        ("net_profit",          "mean_soc_fulfillment","Net Profit (€)",      "Mean SoC Fulfillment", "SoC Fulfilment vs Profit"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, panels):
        if xcol not in agg.columns or ycol not in agg.columns:
            continue
        ax.scatter(agg[xcol], agg[ycol], c=colors[:len(agg)], s=80, zorder=3)
        for _, row in agg.iterrows():
            ax.annotate(row["controller"], (row[xcol], row[ycol]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    save_figure(fig, Path(output_dir) / "tradeoff_scatter.png")


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path) -> None:
    """Net profit vs total_compute_s; Pareto-efficient controllers highlighted.

    A controller is Pareto-efficient if no other controller has both higher
    profit AND lower compute time.
    """
    apply_pub_style()
    agg = df.groupby("controller")[["net_profit", "total_compute_s"]].mean().reset_index()

    # Identify Pareto-efficient points (maximise profit, minimise compute)
    def is_pareto(row):
        return not any(
            (other["net_profit"] >= row["net_profit"] and
             other["total_compute_s"] <= row["total_compute_s"] and
             (other["net_profit"] > row["net_profit"] or
              other["total_compute_s"] < row["total_compute_s"]))
            for _, other in agg.iterrows()
        )
    agg["pareto"] = agg.apply(is_pareto, axis=1)

    palette = colorblind_palette(2)
    fig, ax = plt.subplots()
    for _, row in agg.iterrows():
        color = palette[0] if row["pareto"] else palette[1]
        marker = "*" if row["pareto"] else "o"
        ax.scatter(row["total_compute_s"], row["net_profit"],
                   c=color, marker=marker, s=120, zorder=3)
        ax.annotate(row["controller"], (row["total_compute_s"], row["net_profit"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=9)

    # Sort Pareto points and draw frontier line
    pareto_pts = agg[agg["pareto"]].sort_values("total_compute_s")
    if len(pareto_pts) > 1:
        ax.plot(pareto_pts["total_compute_s"], pareto_pts["net_profit"],
                "--", color=palette[0], linewidth=1, label="Pareto frontier")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=palette[0],
               markersize=10, label="Pareto-efficient"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[1],
               markersize=8, label="Dominated"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Total Compute Time (s)")
    ax.set_ylabel("Net Profit (€)")
    ax.set_title("Pareto Frontier: Profit vs Compute Time")
    save_figure(fig, Path(output_dir) / "pareto_frontier.png")
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "boxplot or violin or tradeoff or pareto" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add distribution, scatter, and Pareto charts to plots.py"
```

---

## Task 7: plots.py — heatmaps and scaling charts

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
from evopt.analysis.plots import (
    plot_correlation_heatmap, plot_profit_heatmap,
    plot_scaling_compute, plot_scaling_profit, plot_scaling_served_customers,
)

def test_plot_correlation_heatmap(tmp_path):
    plot_correlation_heatmap(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "correlation_heatmap.png")

def test_plot_profit_heatmap(tmp_path):
    plot_profit_heatmap(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "profit_heatmap.png")

def test_plot_scaling_compute(tmp_path):
    plot_scaling_compute(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "scaling_compute.png")

def test_plot_scaling_profit(tmp_path):
    plot_scaling_profit(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "scaling_profit.png")

def test_plot_scaling_served_customers(tmp_path):
    plot_scaling_served_customers(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "scaling_served_customers.png")
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "heatmap or scaling" -v 2>&1 | head -15
```

Expected: `ImportError`.

- [ ] **Step 3: Add heatmap and scaling functions to plots.py**

Append to `src/evopt/analysis/plots.py`:

```python
def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Seaborn heatmap of Pearson correlation matrix across all numeric metrics."""
    import seaborn as sns
    apply_pub_style()
    exclude = {"seed", "ports", "horizon"}
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    corr = df[num_cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Pearson Correlation Matrix")
    save_figure(fig, Path(output_dir) / "correlation_heatmap.png")


def plot_profit_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: mean net_profit by controller (rows) × ports (cols)."""
    import seaborn as sns
    apply_pub_style()
    pivot = df.groupby(["controller", "ports"])["net_profit"].mean().unstack("ports")
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.5), max(5, pivot.shape[0])))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlGn",
        linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Mean Net Profit (€) by Controller × Ports")
    ax.set_xlabel("Ports")
    ax.set_ylabel("Controller")
    save_figure(fig, Path(output_dir) / "profit_heatmap.png")


def _plot_scaling(
    df: pd.DataFrame, output_dir: Path,
    metric: str, ylabel: str, filename: str, title: str,
) -> None:
    """Generic scaling chart: metric vs ports, one line per controller."""
    apply_pub_style()
    controllers = sorted(df["controller"].unique())
    palette = colorblind_palette(len(controllers))
    agg = df.groupby(["controller", "ports"])[metric].mean().reset_index()
    fig, ax = plt.subplots()
    for ctrl, color in zip(controllers, palette):
        sub = agg[agg["controller"] == ctrl].sort_values("ports")
        ax.plot(sub["ports"], sub[metric], marker="o", label=ctrl, color=color)
    ax.set_xlabel("Number of Ports")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Controller", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    save_figure(fig, Path(output_dir) / filename)


def plot_scaling_compute(df: pd.DataFrame, output_dir: Path) -> None:
    """Total compute time (s) vs ports, one line per controller."""
    _plot_scaling(df, output_dir, "total_compute_s", "Total Compute Time (s)",
                  "scaling_compute.png", "Compute Time Scaling with Port Count")


def plot_scaling_profit(df: pd.DataFrame, output_dir: Path) -> None:
    """Net profit (€) vs ports, one line per controller."""
    _plot_scaling(df, output_dir, "net_profit", "Net Profit (€)",
                  "scaling_profit.png", "Net Profit Scaling with Port Count")


def plot_scaling_served_customers(df: pd.DataFrame, output_dir: Path) -> None:
    """Served customers vs ports, one line per controller."""
    _plot_scaling(df, output_dir, "served_customers", "Served Customers",
                  "scaling_served_customers.png",
                  "Served Customers Scaling with Port Count")
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "heatmap or scaling" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add heatmap and scaling charts to plots.py"
```

---

## Task 8: plots.py — radar chart (optional)

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_results_display.py`:

```python
from evopt.analysis.plots import plot_radar

def test_plot_radar(tmp_path):
    plot_radar(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "radar.png")
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py::test_plot_radar -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Add radar function to plots.py**

Append to `src/evopt/analysis/plots.py`:

```python
def plot_radar(df: pd.DataFrame, output_dir: Path) -> None:
    """Optional spider chart: normalised metrics per controller.

    Not included in the default report — only generated when explicitly requested.
    Metrics are min-max normalised across controllers so all axes share [0, 1].
    """
    apply_pub_style()
    _RADAR_METRICS = [m for m in [
        "net_profit", "served_customers", "mean_soc_fulfillment",
        "gap_to_best", "mean_step_ms",
    ] if m in df.columns]

    agg = df.groupby("controller")[_RADAR_METRICS].mean()
    # Min-max normalise per metric; invert gap_to_best and mean_step_ms so higher=better
    normed = agg.copy()
    for col in _RADAR_METRICS:
        mn, mx = agg[col].min(), agg[col].max()
        rng = mx - mn if mx != mn else 1.0
        normed[col] = (agg[col] - mn) / rng
    for col in ["gap_to_best", "mean_step_ms"]:
        if col in normed.columns:
            normed[col] = 1.0 - normed[col]

    controllers = list(normed.index)
    metrics = _RADAR_METRICS
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    palette = colorblind_palette(len(controllers))

    for ctrl, color in zip(controllers, palette):
        values = normed.loc[ctrl].tolist() + normed.loc[ctrl].tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=ctrl, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Controller Comparison (Normalised Metrics)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    save_figure(fig, Path(output_dir) / "radar.png")
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py::test_plot_radar -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add optional radar chart to plots.py"
```

---

## Task 9: correlation.py

**Files:**
- Create: `src/evopt/analysis/correlation.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
# ── correlation ─────────────────────────────────────────────────────────────
from evopt.analysis.correlation import (
    compute_correlation, save_correlation_csv, summarise_correlations,
)

def test_compute_correlation_shape():
    df = _ready_df()
    corr = compute_correlation(df)
    assert corr.shape[0] == corr.shape[1]          # square
    assert (corr.values.diagonal() == pytest.approx(1.0)).all()

def test_compute_correlation_excludes_seed_ports():
    df = _ready_df()
    corr = compute_correlation(df)
    assert "seed" not in corr.columns
    assert "ports" not in corr.columns

def test_save_correlation_csv(tmp_path):
    corr = compute_correlation(_ready_df())
    save_correlation_csv(corr, tmp_path)
    assert (tmp_path / "correlation_matrix.csv").exists()

def test_summarise_correlations_markdown(tmp_path):
    corr = compute_correlation(_ready_df())
    text = summarise_correlations(corr)
    assert "positive" in text.lower() or "negative" in text.lower()
    assert "##" in text or "**" in text
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "correlation" -v 2>&1 | head -15
```

Expected: `ImportError`.

- [ ] **Step 3: Implement correlation.py**

```python
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
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "correlation" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/correlation.py tests/test_results_display.py
git commit -m "feat: add evopt.analysis.correlation (Pearson matrix, CSV, MD summary)"
```

---

## Task 10: stats.py

**Files:**
- Create: `src/evopt/analysis/stats.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
# ── stats ───────────────────────────────────────────────────────────────────
from evopt.analysis.stats import (
    compare_controllers, compare_all_controllers,
    save_significance_csv, save_significance_summary,
)

def test_compare_controllers_returns_keys():
    df = _ready_df()
    result = compare_controllers(df, "net_profit", "milp_h1", "milp_h6")
    for key in ["n_pairs", "mean_a", "mean_b", "mean_diff",
                "p_ttest", "p_wilcoxon", "cohen_d", "is_significant"]:
        assert key in result, f"missing key: {key}"

def test_compare_controllers_paired_cohens_d():
    """cohen_d must be mean(diff)/std(diff), not mean_diff/pooled_std."""
    rng = np.random.default_rng(42)
    rows = []
    for seed in range(20):
        base = rng.normal(10, 1)
        rows.append({"controller": "A", "seed": seed, "ports": 3,
                     "experiment_id": "e", "net_profit": base})
        rows.append({"controller": "B", "seed": seed, "ports": 3,
                     "experiment_id": "e", "net_profit": base + 2.0})
    df = pd.DataFrame(rows)
    result = compare_controllers(df, "net_profit", "A", "B")
    diff = np.array([b - a for a, b in zip(
        df[df["controller"]=="A"].sort_values("seed")["net_profit"].values,
        df[df["controller"]=="B"].sort_values("seed")["net_profit"].values,
    )])
    expected_d = abs(diff.mean() / diff.std(ddof=1))
    assert abs(abs(result["cohen_d"]) - expected_d) < 1e-6

def test_compare_controllers_identical():
    df = _ready_df()
    result = compare_controllers(df, "net_profit", "milp_h1", "milp_h1")
    assert result["cohen_d"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_ttest"] == pytest.approx(1.0)

def test_compare_controllers_drops_unmatched_seeds(recwarn):
    df = _ready_df()
    # Add an extra seed for milp_h1 only
    extra = df[df["controller"] == "milp_h1"].iloc[:1].copy()
    extra["seed"] = 999
    df2 = pd.concat([df, extra], ignore_index=True)
    result = compare_controllers(df2, "net_profit", "milp_h1", "milp_h6")
    # Seed 999 has no match in milp_h6 → n_pairs should be original count
    assert result["n_pairs"] == df["seed"].nunique() * df["ports"].nunique()

def test_compare_controllers_uses_experiment_id():
    df = _ready_df()
    # compare_controllers should auto-detect experiment_id column
    result = compare_controllers(df, "net_profit", "milp_h1", "milp_h6")
    assert result["n_pairs"] > 0

def test_compare_all_controllers_shape():
    df = _ready_df()
    result = compare_all_controllers(df, "net_profit")
    controllers = df["controller"].nunique()
    expected_pairs = controllers * (controllers - 1) // 2
    assert len(result) == expected_pairs

def test_save_significance_csv(tmp_path):
    df = _ready_df()
    results = compare_all_controllers(df, "net_profit")
    save_significance_csv(results, tmp_path)
    assert (tmp_path / "table_significance.csv").exists()
    assert (tmp_path / "table_significance.tex").exists()

def test_save_significance_summary(tmp_path):
    df = _ready_df()
    results = compare_all_controllers(df, "net_profit")
    save_significance_summary(results, tmp_path)
    text = (tmp_path / "significance_summary.md").read_text()
    assert "milp" in text.lower() or "controller" in text.lower()
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "compare or significance" -v 2>&1 | head -20
```

Expected: `ImportError`.

- [ ] **Step 3: Implement stats.py**

```python
# src/evopt/analysis/stats.py
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
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "compare or significance" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/stats.py tests/test_results_display.py
git commit -m "feat: add evopt.analysis.stats with paired t-test, Wilcoxon, and paired Cohen's d"
```

---

## Task 11: report.py

**Files:**
- Create: `src/evopt/analysis/report.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_results_display.py`:

```python
# ── report ──────────────────────────────────────────────────────────────────
from evopt.analysis.report import write_markdown_report, write_html_report
from evopt.analysis.tables import (
    make_full_table, make_best_controller_table, make_tradeoff_table,
    make_robustness_table,
)
from evopt.analysis.correlation import compute_correlation, summarise_correlations
from evopt.analysis.stats import compare_all_controllers, save_significance_summary

def _make_tables(df, tmp_path):
    return {
        "full":       make_full_table(df, tmp_path),
        "best":       make_best_controller_table(df, tmp_path),
        "tradeoff":   make_tradeoff_table(df, tmp_path),
        "robustness": make_robustness_table(df, tmp_path),
    }

def test_write_markdown_report(tmp_path):
    df = _ready_df()
    tables = _make_tables(df, tmp_path)
    corr = compute_correlation(df)
    corr_summary = summarise_correlations(corr)
    sig_results = compare_all_controllers(df, "net_profit")
    save_significance_summary(sig_results, tmp_path)
    sig_summary = (tmp_path / "significance_summary.md").read_text()
    write_markdown_report(df, tables, corr_summary, sig_summary, tmp_path, "exp_a")
    text = (tmp_path / "benchmark_report.md").read_text()
    assert "## Experiment Configuration" in text
    assert "## Key Findings" in text
    assert "## Robustness" in text

def test_report_warns_multi_experiment(tmp_path):
    df1 = _sample_df(); df1["experiment_id"] = "exp_a"
    df2 = _sample_df(); df2["experiment_id"] = "exp_b"
    combined = add_derived_metrics(clean_results(
        pd.concat([df1, df2], ignore_index=True)
    ))
    tables = _make_tables(combined, tmp_path)
    corr_summary = summarise_correlations(compute_correlation(combined))
    write_markdown_report(combined, tables, corr_summary, "", tmp_path, "multi")
    text = (tmp_path / "benchmark_report.md").read_text()
    assert "Warning" in text and "experiment" in text.lower()

def test_write_html_report(tmp_path):
    df = _ready_df()
    tables = _make_tables(df, tmp_path)
    # Create placeholder PNGs so embed logic doesn't crash
    for name in ["profit_by_controller", "profit_vs_horizon", "compute_vs_horizon",
                 "profit_boxplot", "profit_violin", "compute_boxplot",
                 "tradeoff_scatter", "pareto_frontier", "correlation_heatmap",
                 "profit_heatmap", "scaling_compute", "scaling_profit",
                 "scaling_served_customers"]:
        (tmp_path / f"{name}.png").write_bytes(b"\x89PNG\r\n")
    write_html_report(df, tables, tmp_path, "exp_a")
    html = (tmp_path / "benchmark_report.html").read_text()
    assert "<html" in html
    assert "benchmark" in html.lower()
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "report" -v 2>&1 | head -15
```

Expected: `ImportError`.

- [ ] **Step 3: Implement report.py**

```python
# src/evopt/analysis/report.py
"""Generate benchmark_report.md and benchmark_report.html."""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd


def _df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    return df.head(max_rows).to_markdown(floatfmt=".3f") if hasattr(df, "to_markdown") else df.head(max_rows).to_string()


def _key_findings(df: pd.DataFrame) -> str:
    lines = []
    try:
        mean_profit = df.groupby("controller")["net_profit"].mean()
        best_ctrl = mean_profit.idxmax()
        best_val = mean_profit.max()
        lines.append(f"- **Best net_profit:** `{best_ctrl}` with mean **€{best_val:.2f}**")
    except Exception:
        pass
    try:
        mean_ms = df.groupby("controller")["mean_step_ms"].mean()
        fastest = mean_ms.idxmin()
        lines.append(f"- **Lowest compute time:** `{fastest}` at {mean_ms.min():.2f} ms/step")
    except Exception:
        pass
    try:
        if "profit_per_compute_s" in df.columns:
            eff = df.groupby("controller")["profit_per_compute_s"].mean()
            best_eff = eff.idxmax()
            lines.append(f"- **Best efficiency (€/compute-s):** `{best_eff}` = {eff.max():.3f}")
    except Exception:
        pass
    return "\n".join(lines) if lines else "No findings computed."


def write_markdown_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    corr_summary: str,
    sig_summary: str,
    output_dir: Path,
    exp_id: str,
) -> None:
    """Write benchmark_report.md with all analysis sections."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_exp = df["experiment_id"].unique().tolist() if "experiment_id" in df.columns else [exp_id]
    multi_warn = ""
    if len(unique_exp) > 1:
        multi_warn = (
            f"\n> **Warning: this report contains data from {len(unique_exp)} distinct "
            f"experiments:** {', '.join(str(e) for e in unique_exp)}. "
            "Tables and charts aggregate across all of them. "
            "Filter by experiment_id before drawing per-experiment conclusions.\n"
        )

    # Config snapshot from first row
    config_rows = []
    for col in ["tariff", "bess_enabled", "v2g_enabled", "solver", "ports"]:
        if col in df.columns:
            val = df[col].iloc[0] if col != "ports" else sorted(df["ports"].unique().tolist())
            config_rows.append(f"| {col} | {val} |")
    config_table = "| Setting | Value |\n|---|---|\n" + "\n".join(config_rows)

    sections = [
        f"# Benchmark Report — `{exp_id}`\n",
        multi_warn,
        "## Experiment Configuration\n",
        config_table + "\n",
        f"Experiment IDs: {', '.join(str(e) for e in unique_exp)}\n",
        "## Key Findings\n",
        _key_findings(df) + "\n",
        "## Best-Performing Controller\n",
        _df_to_md(tables.get("best", pd.DataFrame())) + "\n",
        "## Full Results Table\n",
        _df_to_md(tables.get("full", pd.DataFrame())) + "\n",
        "## Trade-off Analysis\n",
        _df_to_md(tables.get("tradeoff", pd.DataFrame())) + "\n",
        "## Scalability Analysis\n",
        "![Profit Scaling](scaling_profit.png)\n",
        "![Compute Scaling](scaling_compute.png)\n",
        "![Served Customers Scaling](scaling_served_customers.png)\n",
        "## Statistical Significance Analysis\n",
        sig_summary + "\n",
        "## Correlation Analysis\n",
        corr_summary + "\n",
        "## Robustness Analysis\n",
        _df_to_md(tables.get("robustness", pd.DataFrame())) + "\n",
    ]

    (output_dir / "benchmark_report.md").write_text("\n".join(sections))


def write_html_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    exp_id: str,
) -> None:
    """Write a self-contained benchmark_report.html with embedded PNG images."""
    output_dir = Path(output_dir)

    def _embed(filename: str) -> str:
        p = output_dir / filename
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'

    def _tbl(key: str) -> str:
        tbl = tables.get(key, pd.DataFrame())
        if tbl.empty:
            return "<p><em>No data</em></p>"
        return tbl.to_html(classes="table", border=0, float_format=lambda x: f"{x:.3f}")

    unique_exp = df["experiment_id"].unique().tolist() if "experiment_id" in df.columns else [exp_id]
    multi_warn_html = ""
    if len(unique_exp) > 1:
        multi_warn_html = (
            f'<p style="color:orange;font-weight:bold">Warning: {len(unique_exp)} '
            f"experiment IDs present: {', '.join(str(e) for e in unique_exp)}. "
            "Filter by experiment_id before drawing per-experiment conclusions.</p>"
        )

    charts = [
        "profit_by_controller.png", "profit_vs_horizon.png", "compute_vs_horizon.png",
        "profit_boxplot.png", "profit_violin.png", "compute_boxplot.png",
        "tradeoff_scatter.png", "pareto_frontier.png",
        "correlation_heatmap.png", "profit_heatmap.png",
        "scaling_profit.png", "scaling_compute.png", "scaling_served_customers.png",
    ]

    css = (
        "body{font-family:sans-serif;max-width:1100px;margin:auto;padding:20px}"
        "h1,h2{color:#333}table{border-collapse:collapse;width:100%;font-size:13px}"
        "td,th{border:1px solid #ddd;padding:6px 10px}th{background:#f2f2f2}"
        "tr:nth-child(even){background:#fafafa}"
    )

    html_parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Benchmark Report — {exp_id}</title>"
        f"<style>{css}</style></head><body>",
        f"<h1>Benchmark Report — <code>{exp_id}</code></h1>",
        multi_warn_html,
        "<h2>Key Findings</h2>", f"<pre>{_key_findings(df)}</pre>",
        "<h2>Best-Performing Controller</h2>", _tbl("best"),
        "<h2>Full Results Table</h2>", _tbl("full"),
        "<h2>Trade-off Analysis</h2>", _tbl("tradeoff"),
        "<h2>Robustness Analysis</h2>", _tbl("robustness"),
        "<h2>Charts</h2>",
    ]
    for chart in charts:
        html_parts.append(_embed(chart))
    html_parts.append("</body></html>")

    (output_dir / "benchmark_report.html").write_text("\n".join(html_parts))
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "report" -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/report.py tests/test_results_display.py
git commit -m "feat: add benchmark_report.md and self-contained HTML report"
```

---

## Task 12: __main__.py (CLI)

**Files:**
- Create: `src/evopt/analysis/__main__.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_results_display.py`:

```python
# ── CLI ─────────────────────────────────────────────────────────────────────
import subprocess, sys

def test_default_output_dir(tmp_path):
    """When --output is omitted, outputs land in input_path.parent/analysis/."""
    df = _sample_df()
    csv_path = tmp_path / "bench.csv"
    df.to_csv(csv_path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "evopt.analysis", "--input", str(csv_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "analysis" / "benchmark_report.md").exists()

def test_explicit_output_dir(tmp_path):
    df = _sample_df()
    csv_path = tmp_path / "bench.csv"
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "evopt.analysis",
         "--input", str(csv_path), "--output", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "benchmark_report.md").exists()
```

- [ ] **Step 2: Run — confirm failure**

```
python -m pytest tests/test_results_display.py -k "output_dir" -v 2>&1 | head -15
```

Expected: error (no `__main__.py` yet).

- [ ] **Step 3: Implement __main__.py**

```python
# src/evopt/analysis/__main__.py
"""CLI entry point for the EV benchmark analysis pipeline.

Usage:
    python -m evopt.analysis --input results/benchmark.csv
    python -m evopt.analysis --input results/benchmark.csv --output results/figures
    python -m evopt.analysis --input results/benchmark.csv --no-radar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tables, charts, and reports from an EV benchmark CSV."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to benchmark CSV or JSON file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <input_dir>/analysis/)")
    parser.add_argument("--no-radar", action="store_true",
                        help="Skip the optional radar chart")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    from evopt.analysis import correlation, plots, report, stats, tables
    from evopt.analysis.loader import add_derived_metrics, clean_results, load_results

    print(f"Loading {input_path} …")
    df = add_derived_metrics(clean_results(load_results(input_path)))
    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "benchmark"

    print("Generating tables …")
    tbls = {
        "full":       tables.make_full_table(df, output_dir),
        "cross_port": tables.make_cross_port_table(df, output_dir),
        "best":       tables.make_best_controller_table(df, output_dir),
        "compute":    tables.make_compute_table(df, output_dir),
        "efficiency": tables.make_efficiency_table(df, output_dir),
        "tradeoff":   tables.make_tradeoff_table(df, output_dir),
        "robustness": tables.make_robustness_table(df, output_dir),
    }

    print("Generating charts …")
    plots.plot_profit_by_controller(df, output_dir)
    plots.plot_profit_vs_horizon(df, output_dir)
    plots.plot_compute_vs_horizon(df, output_dir)
    plots.plot_profit_boxplot(df, output_dir)
    plots.plot_profit_violin(df, output_dir)
    plots.plot_compute_boxplot(df, output_dir)
    plots.plot_tradeoff_scatter(df, output_dir)
    plots.plot_pareto_frontier(df, output_dir)
    plots.plot_correlation_heatmap(df, output_dir)
    plots.plot_profit_heatmap(df, output_dir)
    plots.plot_scaling_compute(df, output_dir)
    plots.plot_scaling_profit(df, output_dir)
    plots.plot_scaling_served_customers(df, output_dir)
    if not args.no_radar:
        plots.plot_radar(df, output_dir)

    print("Computing correlation …")
    corr = correlation.compute_correlation(df)
    correlation.save_correlation_csv(corr, output_dir)
    corr_summary = correlation.summarise_correlations(corr)

    print("Running significance tests …")
    sig_results = stats.compare_all_controllers(df, "net_profit")
    stats.save_significance_csv(sig_results, output_dir)
    stats.save_significance_summary(sig_results, output_dir)
    sig_summary = (output_dir / "significance_summary.md").read_text()

    print("Writing reports …")
    report.write_markdown_report(df, tbls, corr_summary, sig_summary, output_dir, exp_id)
    report.write_html_report(df, tbls, output_dir, exp_id)

    # Summary of outputs
    files = sorted(output_dir.iterdir())
    print(f"\nOutputs written to {output_dir}/")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — confirm pass**

```
python -m pytest tests/test_results_display.py -k "output_dir" -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/test_results_display.py -v
```

Expected: all tests green.

- [ ] **Step 6: Commit**

```
git add src/evopt/analysis/__main__.py tests/test_results_display.py
git commit -m "feat: add evopt.analysis CLI (__main__.py) with default output dir"
```

---

## Task 13: Modify run_chargax_benchmark.py

**Files:**
- Modify: `src/evopt/experiments/run_chargax_benchmark.py`

No new tests needed — the existing benchmark test suite (`src/evopt/tests/`) covers the benchmark runner. The `--save` addition is additive and does not change existing behaviour when the flag is omitted.

- [ ] **Step 1: Read the existing file before editing**

Open `src/evopt/experiments/run_chargax_benchmark.py` and locate:
- The `argparse` block starting at line 230
- The `main()` function signature and body (lines 147–228)
- The per-port loop starting around line 187

- [ ] **Step 2: Add imports at the top of the file**

After the existing imports (after line 28, `from evopt.experiments.station_configs import ...`), add:

```python
import sys
from evopt.analysis.utils import format_experiment_id, save_metadata
```

- [ ] **Step 3: Add `--save` argument to argparse block**

In the `if __name__ == "__main__":` block, after the `--no-must-serve` argument (around line 247), add:

```python
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Save raw per-seed results as CSV to PATH (e.g. results/benchmark.csv). "
             "A metadata.json sidecar is written to the same directory.",
    )
```

- [ ] **Step 4: Thread `save` and config through to main()**

Change the `main()` signature to add `save_path` and `cli_command` parameters:

```python
def main(
    n_seeds:                int              = 10,
    ev_tariff:              float | None     = 0.75,
    ev_markup:              float | None     = None,
    horizons:               list[int] | None = None,
    ports:                  list[int] | None = None,
    use_bess:               bool             = True,
    allow_discharging:      bool             = False,
    allow_bess_discharging: bool             = False,
    must_serve:             bool             = True,
    save_path:              str | None       = None,
    cli_command:            str              = "",
) -> None:
```

- [ ] **Step 5: Collect raw per-port DataFrames inside main()**

At the start of `main()` (before the `for n_ports in ports:` loop), add:

```python
    if horizons is None:
        horizons = [12]
    if ports is None:
        ports = [3]

    tariff_str = (
        f"dynamic:{ev_markup}" if ev_markup is not None
        else "dynamic" if ev_tariff is None
        else str(ev_tariff)
    )
    experiment_id = format_experiment_id(
        ports=ports, horizons=horizons, tariff=tariff_str,
    )

    per_port_raw_dfs: list[pd.DataFrame] = []   # ← add this line
```

(The existing `if horizons is None` and `if ports is None` blocks are already there — only add `per_port_raw_dfs`, `tariff_str`, and `experiment_id`.)

- [ ] **Step 6: Stamp config columns on each per-port DataFrame**

Inside the `for n_ports in ports:` loop, after the line `df = _run_one_config(...)` (around line 203), add:

```python
        # Stamp config columns for structured export
        raw = df.copy()
        raw["ports"]         = n_ports
        raw["experiment_id"] = experiment_id
        raw["tariff"]        = tariff_str
        raw["bess_enabled"]  = use_bess
        raw["v2g_enabled"]   = allow_discharging and use_bess
        raw["solver"]        = "highs"
        per_port_raw_dfs.append(raw)
```

- [ ] **Step 7: Save CSV and metadata after the port loop**

After the existing cross-port summary printing block (after the `if len(ports) > 1:` block), add:

```python
    if save_path is not None:
        from pathlib import Path
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        combined_raw = pd.concat(per_port_raw_dfs, ignore_index=True)
        combined_raw.to_csv(save_p, index=False)
        print(f"\nResults saved to {save_p}")
        save_metadata(
            config={
                "experiment_id":  experiment_id,
                "cli_command":    cli_command,
                "ports":          ports,
                "horizons":       horizons,
                "n_seeds":        n_seeds,
                "tariff":         tariff_str,
                "bess_enabled":   use_bess,
                "v2g_enabled":    allow_discharging and use_bess,
                "solver":         "highs",
            },
            output_dir=save_p.parent,
        )
```

- [ ] **Step 8: Pass args.save and cli_command in the __main__ block**

In the `if __name__ == "__main__":` block, update the `main(...)` call:

```python
    main(
        n_seeds=args.seeds,
        ev_tariff=ev_tariff,
        ev_markup=ev_markup,
        horizons=args.horizons,
        use_bess=not args.no_bess,
        allow_discharging=args.allow_discharging,
        allow_bess_discharging=args.allow_bess_discharging,
        ports=args.ports,
        must_serve=not args.no_must_serve,
        save_path=args.save,
        cli_command=" ".join(sys.argv),
    )
```

- [ ] **Step 9: Smoke-test the import**

```
python -c "from evopt.experiments.run_chargax_benchmark import main; print('OK')"
```

Expected: `OK`.

- [ ] **Step 10: Verify --help shows --save**

```
python -m evopt.experiments.run_chargax_benchmark --help
```

Expected: output includes `--save PATH`.

- [ ] **Step 11: Commit**

```
git add src/evopt/experiments/run_chargax_benchmark.py
git commit -m "feat: add --save flag to run_chargax_benchmark for structured CSV export"
```

---

## Task 14: Full test suite + final check

- [ ] **Step 1: Run the complete test suite**

```
python -m pytest tests/ src/evopt/tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all green. If any pre-existing tests regress, investigate before proceeding.

- [ ] **Step 2: Smoke-test the full pipeline end-to-end**

```python
# Run this as a quick script to verify the pipeline works on synthetic data
python -c "
import pandas as pd, numpy as np, tempfile, subprocess, sys
from pathlib import Path

rng = np.random.default_rng(0)
rows = []
for ctrl, base in [('milp_h1',8),('milp_h6',12),('equal_share',5)]:
    for ports in [3,6]:
        for seed in range(5):
            rows.append({'experiment_id':'smoke','controller':ctrl,'seed':seed,
                'ports':ports,'tariff':'0.75','bess_enabled':True,'v2g_enabled':False,
                'solver':'highs','net_profit':rng.normal(base,1.5),
                'total_revenue':rng.normal(base+10,2),'total_cost':rng.normal(10,0.5),
                'served_customers':float(rng.integers(3,9)),
                'rejected_customers':float(rng.integers(0,3)),
                'mean_soc_fulfillment':rng.uniform(0.7,1.0),'gap_to_best':0.0,
                'total_compute_s':0.0 if ctrl=='equal_share' else rng.uniform(0.1,2.0),
                'mean_step_ms':0.0 if ctrl=='equal_share' else rng.uniform(1,50)})
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / 'bench.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    r = subprocess.run([sys.executable,'-m','evopt.analysis','--input',str(csv)],
                       capture_output=True,text=True)
    print(r.stdout[-500:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-500:])
        sys.exit(1)
    out = csv.parent / 'analysis'
    for f in ['benchmark_report.md','benchmark_report.html','table_full.csv',
              'profit_by_controller.png','correlation_matrix.csv']:
        assert (out/f).exists(), f'Missing: {f}'
    print('Smoke test PASSED')
"
```

Expected: `Smoke test PASSED`.

- [ ] **Step 3: Commit**

```
git add .
git commit -m "test: verify full analysis pipeline passes end-to-end smoke test"
```

---

## Self-review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `utils.py` — all 7 functions | Task 2 |
| `loader.py` — load/clean/derive | Task 3 |
| `tables.py` — 7 tables, CSV + LaTeX | Task 4 |
| Profit charts with 95% CI error bars | Task 5 |
| Distribution charts (boxplot, violin) | Task 6 |
| Scatter + Pareto frontier | Task 6 |
| Heatmaps (correlation, profit) | Task 7 |
| Scaling charts (compute, profit, served_customers) | Task 7 |
| Radar chart (optional) | Task 8 |
| `correlation.py` | Task 9 |
| `stats.py` — paired t-test, Wilcoxon, paired Cohen's d | Task 10 |
| `report.py` — MD + HTML, multi-experiment warning | Task 11 |
| `__main__.py` — default output dir, --no-radar | Task 12 |
| `run_chargax_benchmark.py` — --save, metadata.json | Task 13 |
| gap_to_best recomputed within (exp_id, seed, ports) | Task 3 |
| experiment_id in raw CSV rows | Task 13 |
| tariff/bess_enabled/v2g_enabled/solver in raw CSV | Task 13 |
| profit_per_compute_s NaN for zero-compute | Task 3 |
| match_on auto-includes experiment_id | Task 10 |
| All tests from spec Section 6 | Tasks 2–12 |
| pyproject.toml deps | Task 1 |

All spec requirements covered. No placeholders, no TBDs.
