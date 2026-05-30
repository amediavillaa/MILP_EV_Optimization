# Horizon Comparison Figures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `plot_horizon_comparison` (2-panel) and `plot_horizon_full` (2×2 panel) to `evopt.analysis.plots`, both filtering to MILP-only data, and wire them into the analysis pipeline CLI.

**Architecture:** One private helper `_horizon_agg` handles MILP filtering + CI aggregation for any metric. The two public functions compose panels from that helper. Both are called unconditionally in `__main__.py`; they return early silently when no MILP data exists.

**Tech Stack:** Matplotlib, NumPy, Pandas; existing `apply_pub_style`, `colorblind_palette`, `save_figure` from `evopt.analysis.utils`.

**Spec:** `docs/superpowers/specs/2026-05-30-horizon-comparison-figures-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/evopt/analysis/plots.py` | Add `_horizon_agg`, `plot_horizon_comparison`, `plot_horizon_full` |
| Modify | `src/evopt/analysis/__main__.py` | Call both new functions in the pipeline |
| Modify | `tests/test_results_display.py` | Tests for helper and both figure functions |

---

## Task 1: `_horizon_agg` helper + tests

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Add failing tests for `_horizon_agg`**

Add to `tests/test_results_display.py` (after the existing imports and `_sample_df`):

```python
import math

def _milp_df(n_seeds: int = 3) -> pd.DataFrame:
    """DataFrame with MILP-only controllers (milp_h1, milp_h3, milp_h6, milp_h12)."""
    rng = np.random.default_rng(42)
    rows = []
    for horizon in [1, 3, 6, 12]:
        ctrl = f"milp_h{horizon}"
        for ports in [3, 6]:
            for seed in range(n_seeds):
                rows.append({
                    "controller":           ctrl,
                    "horizon":              float(horizon),
                    "ports":                ports,
                    "seed":                 seed,
                    "net_profit":           rng.normal(8.0 * horizon, 1.0),
                    "mean_step_ms":         abs(rng.normal(20.0 * horizon, 3.0)),
                    "served_customers":     float(rng.integers(3, 9)),
                    "mean_soc_fulfillment": rng.uniform(0.7, 1.0),
                    "total_compute_s":      abs(rng.normal(2.0 * horizon, 0.5)),
                })
    return pd.DataFrame(rows)


# ── _horizon_agg ────────────────────────────────────────────────────────────
from evopt.analysis.plots import _horizon_agg


def test_horizon_agg_returns_expected_columns():
    df = _milp_df()
    agg = _horizon_agg(df, "net_profit")
    assert {"horizon", "ports", "mean", "std", "n", "ci"}.issubset(agg.columns)


def test_horizon_agg_excludes_non_milp_rows():
    df = _milp_df()
    # Add a baseline controller row (horizon = NaN)
    baseline = df.iloc[0].copy()
    baseline["controller"] = "equal_share"
    baseline["horizon"] = float("nan")
    df = pd.concat([df, baseline.to_frame().T], ignore_index=True)
    agg = _horizon_agg(df, "net_profit")
    assert agg["n"].min() >= 1
    assert len(agg) == len([1, 3, 6, 12]) * len([3, 6])  # 4 horizons × 2 port vals


def test_horizon_agg_returns_empty_when_no_milp():
    df = _milp_df()
    df["horizon"] = float("nan")   # make all rows non-MILP
    agg = _horizon_agg(df, "net_profit")
    assert agg.empty


def test_horizon_agg_returns_empty_for_missing_metric():
    df = _milp_df()
    agg = _horizon_agg(df, "nonexistent_column")
    assert agg.empty
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_results_display.py::test_horizon_agg_returns_expected_columns -v
```

Expected: FAIL — `cannot import name '_horizon_agg'`

- [ ] **Step 3: Implement `_horizon_agg` in `plots.py`**

Add after the existing imports in `src/evopt/analysis/plots.py`, before the first public function:

```python
def _horizon_agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate *metric* by (horizon, ports) for MILP-only rows.

    Returns an empty DataFrame when no MILP data exists or the metric
    column is absent.
    """
    milp = df[df["horizon"].notna()].copy()
    if milp.empty or metric not in milp.columns:
        return pd.DataFrame()
    agg = (
        milp.groupby(["horizon", "ports"])[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
    return agg
```

- [ ] **Step 4: Run all four tests — expect 4 passed**

```
pytest tests/test_results_display.py::test_horizon_agg_returns_expected_columns tests/test_results_display.py::test_horizon_agg_excludes_non_milp_rows tests/test_results_display.py::test_horizon_agg_returns_empty_when_no_milp tests/test_results_display.py::test_horizon_agg_returns_empty_for_missing_metric -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add _horizon_agg helper for MILP-only CI aggregation"
```

---

## Task 2: `plot_horizon_comparison` + tests

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Add failing tests**

Add to `tests/test_results_display.py`:

```python
# ── plot_horizon_comparison ─────────────────────────────────────────────────
from evopt.analysis.plots import plot_horizon_comparison


def test_plot_horizon_comparison_creates_png(tmp_path):
    df = _milp_df()
    plot_horizon_comparison(df, tmp_path)
    assert (tmp_path / "horizon_comparison.png").exists()


def test_plot_horizon_comparison_no_file_when_no_milp(tmp_path):
    df = _milp_df()
    df["horizon"] = float("nan")
    plot_horizon_comparison(df, tmp_path)
    assert not (tmp_path / "horizon_comparison.png").exists()
```

- [ ] **Step 2: Run tests — expect ImportError / AttributeError**

```
pytest tests/test_results_display.py::test_plot_horizon_comparison_creates_png -v
```

Expected: FAIL — `cannot import name 'plot_horizon_comparison'`

- [ ] **Step 3: Implement `plot_horizon_comparison`**

Add to `src/evopt/analysis/plots.py` (after `plot_compute_vs_horizon`):

```python
def plot_horizon_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Two-panel figure comparing MILP horizons: net profit (left) and
    mean step time on a log scale (right). Lines are grouped by port count.

    Skips silently when no MILP data is present.
    """
    apply_pub_style()
    agg_profit = _horizon_agg(df, "net_profit")
    if agg_profit.empty:
        return

    ports_vals = sorted(agg_profit["ports"].unique())
    palette = colorblind_palette(len(ports_vals))
    agg_compute = _horizon_agg(df, "mean_step_ms")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: net profit with 95% CI fill bands
    for port, color in zip(ports_vals, palette):
        sub = agg_profit[agg_profit["ports"] == port].sort_values("horizon")
        ax_l.plot(sub["horizon"], sub["mean"], marker="o", color=color,
                  label=f"{port} ports")
        ax_l.fill_between(
            sub["horizon"],
            sub["mean"] - sub["ci"],
            sub["mean"] + sub["ci"],
            alpha=0.2, color=color,
        )
    ax_l.set_xlabel("Horizon (steps)")
    ax_l.set_ylabel("Net Profit (€)")

    # Right panel: mean step time, log scale, CI error bars
    if not agg_compute.empty:
        for port, color in zip(ports_vals, palette):
            sub = agg_compute[agg_compute["ports"] == port].sort_values("horizon")
            ax_r.errorbar(
                sub["horizon"], sub["mean"], yerr=sub["ci"],
                marker="s", capsize=4, color=color, label=f"{port} ports",
            )
    ax_r.set_yscale("log")
    ax_r.set_xlabel("Horizon (steps)")
    ax_r.set_ylabel("Mean Step Time (ms)")

    # Shared legend outside the right panel
    ax_r.legend(title="Ports", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    save_figure(fig, Path(output_dir) / "horizon_comparison.png")
```

- [ ] **Step 4: Run all tests — expect 2 new + all prior passing**

```
pytest tests/test_results_display.py -v -k "horizon_comparison or horizon_agg"
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add plot_horizon_comparison (profit + compute, MILP-only)"
```

---

## Task 3: `plot_horizon_full` + tests

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Add failing tests**

Add to `tests/test_results_display.py`:

```python
# ── plot_horizon_full ───────────────────────────────────────────────────────
from evopt.analysis.plots import plot_horizon_full


def test_plot_horizon_full_creates_png(tmp_path):
    df = _milp_df()
    plot_horizon_full(df, tmp_path)
    assert (tmp_path / "horizon_full.png").exists()


def test_plot_horizon_full_no_file_when_no_milp(tmp_path):
    df = _milp_df()
    df["horizon"] = float("nan")
    plot_horizon_full(df, tmp_path)
    assert not (tmp_path / "horizon_full.png").exists()


def test_plot_horizon_full_handles_missing_metric(tmp_path):
    df = _milp_df().drop(columns=["mean_soc_fulfillment"])
    plot_horizon_full(df, tmp_path)
    # Should still create the figure with the remaining 3 panels
    assert (tmp_path / "horizon_full.png").exists()
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_results_display.py::test_plot_horizon_full_creates_png -v
```

Expected: FAIL — `cannot import name 'plot_horizon_full'`

- [ ] **Step 3: Implement `plot_horizon_full`**

Add to `src/evopt/analysis/plots.py` (after `plot_horizon_comparison`):

```python
def plot_horizon_full(df: pd.DataFrame, output_dir: Path) -> None:
    """2×2 panel figure: net profit, mean step time (log), served customers,
    and mean SoC fulfillment vs MILP horizon. Lines grouped by port count.

    Panels whose metric column is absent from df are hidden. Skips silently
    when no MILP data is present.
    """
    apply_pub_style()
    agg_profit = _horizon_agg(df, "net_profit")
    if agg_profit.empty:
        return

    ports_vals = sorted(agg_profit["ports"].unique())
    palette = colorblind_palette(len(ports_vals))

    _PANELS = [
        ("net_profit",           "Net Profit (€)",        "linear"),
        ("mean_step_ms",         "Mean Step Time (ms)",   "log"),
        ("served_customers",     "Served Customers",      "linear"),
        ("mean_soc_fulfillment", "Mean SoC Fulfillment",  "linear"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    legend_handle_ax = None
    for ax, (metric, ylabel, scale) in zip(axes_flat, _PANELS):
        agg = _horizon_agg(df, metric)
        if agg.empty:
            ax.set_visible(False)
            continue
        for port, color in zip(ports_vals, palette):
            sub = agg[agg["ports"] == port].sort_values("horizon")
            if scale == "log":
                ax.errorbar(
                    sub["horizon"], sub["mean"], yerr=sub["ci"],
                    marker="s", capsize=4, color=color, label=f"{port} ports",
                )
            else:
                ax.plot(sub["horizon"], sub["mean"], marker="o",
                        color=color, label=f"{port} ports")
                ax.fill_between(
                    sub["horizon"],
                    sub["mean"] - sub["ci"],
                    sub["mean"] + sub["ci"],
                    alpha=0.2, color=color,
                )
        ax.set_yscale(scale)
        ax.set_xlabel("Horizon (steps)")
        ax.set_ylabel(ylabel)
        legend_handle_ax = ax

    # Place single legend outside the last visible panel
    if legend_handle_ax is not None:
        handles, labels = legend_handle_ax.get_legend_handles_labels()
        legend_handle_ax.legend(
            handles, labels, title="Ports",
            bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9,
        )

    save_figure(fig, Path(output_dir) / "horizon_full.png")
```

- [ ] **Step 4: Run all horizon tests — expect 9 passed**

```
pytest tests/test_results_display.py -v -k "horizon"
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/analysis/plots.py tests/test_results_display.py
git commit -m "feat: add plot_horizon_full (2x2 panel, MILP-only)"
```

---

## Task 4: Wire into pipeline + integration test

**Files:**
- Modify: `src/evopt/analysis/__main__.py`
- Modify: `tests/test_results_display.py`

- [ ] **Step 1: Add integration test**

Add to `tests/test_results_display.py`:

```python
# ── pipeline integration ────────────────────────────────────────────────────
import subprocess, sys

def test_pipeline_generates_horizon_figures(tmp_path):
    """End-to-end: analysis CLI produces both horizon figure files."""
    csv_path = tmp_path / "bench.csv"
    # Write a minimal CSV that the loader can parse
    df = _milp_df()
    df["experiment_id"] = "test"
    df["tariff"] = "fixed"
    df["bess_enabled"] = False
    df["v2g_enabled"] = False
    df["solver"] = "highs"
    df["total_revenue"] = df["net_profit"] + 5.0
    df["total_cost"] = 5.0
    df["rejected_customers"] = 0.0
    df["gap_to_best"] = 0.0
    df["total_compute_s"] = df["mean_step_ms"] / 1000.0
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "evopt.analysis",
         "--input", str(csv_path), "--output", str(out_dir), "--no-radar"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "horizon_comparison.png").exists()
    assert (out_dir / "horizon_full.png").exists()
```

- [ ] **Step 2: Run test — expect failure (functions not called yet)**

```
pytest tests/test_results_display.py::test_pipeline_generates_horizon_figures -v
```

Expected: FAIL — files not produced.

- [ ] **Step 3: Add both calls to `__main__.py`**

In `src/evopt/analysis/__main__.py`, add two lines after the existing `plot_compute_vs_horizon` call (line 56):

```python
    plots.plot_compute_vs_horizon(df, output_dir)
    plots.plot_horizon_comparison(df, output_dir)   # ← add
    plots.plot_horizon_full(df, output_dir)          # ← add
    plots.plot_profit_boxplot(df, output_dir)
```

- [ ] **Step 4: Run full test suite — all tests pass**

```
pytest tests/test_results_display.py -v
```

Expected: all pass (including the new integration test).

- [ ] **Step 5: Regenerate dynamic1_3 figures to verify outputs**

```
python -m evopt.analysis --input results/dynamic1_3_discharging_tariff.csv --output results/dynamic1_3_discharging_analysis
```

Expected output includes:
```
  horizon_comparison.png
  horizon_full.png
```

- [ ] **Step 6: Commit**

```bash
git add src/evopt/analysis/__main__.py tests/test_results_display.py
git commit -m "feat: wire plot_horizon_comparison and plot_horizon_full into analysis pipeline"
```
