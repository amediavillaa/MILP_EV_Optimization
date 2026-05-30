# Design: Horizon Comparison Figures

**Date:** 2026-05-30
**Status:** Approved

## Goal

Add two new publication figures to the analysis pipeline that compare MILP horizon variants (milp_h1, milp_h3, milp_h6, milp_h12) directly, without baseline controllers. These isolate the effect of horizon length on performance.

## Context

The existing `plot_profit_vs_horizon` and `plot_compute_vs_horizon` include all controllers and use lines-per-port as the grouping. The new figures filter to MILP-only and show cleaner horizon-focused comparisons with the log scale on compute already applied.

## Implementation

**Files modified:**
- `src/evopt/analysis/plots.py` — two new functions added
- `src/evopt/analysis/__main__.py` — two new calls added to the pipeline

**Data filtering:** rows where `df["horizon"].notna()` (MILP controllers only). The `horizon` column is populated by `add_derived_metrics` via `extract_horizon()` in `utils.py`.

---

## Figure 1 — `horizon_comparison.png`

**Function:** `plot_horizon_comparison(df: pd.DataFrame, output_dir: Path) -> None`

Two panels side by side. Figure size: `(10, 4)`.

### Left panel — Net Profit

- x: horizon value (integer), ticks at each unique horizon
- y: mean net profit (€), 95% CI fill bands
- One line per distinct `ports` value, colorblind palette
- y-axis label: "Net Profit (€)"

### Right panel — Mean Step Time

- Same x-axis structure
- y: mean `mean_step_ms`, log scale, 95% CI error bars (`ax.errorbar` with `capsize=4`)
- y-axis label: "Mean Step Time (ms)"
- `ax.set_yscale("log")`

### Shared elements

- x-axis label: "Horizon (steps)" on both panels
- Single shared legend (lines per port count) placed outside the right panel: `bbox_to_anchor=(1.02, 1), loc="upper left"`
- No titles on individual panels (figure caption in the paper carries the description)

---

## Figure 2 — `horizon_full.png`

**Function:** `plot_horizon_full(df: pd.DataFrame, output_dir: Path) -> None`

2×2 panel grid, shared x-axis semantics. Figure size: `(10, 8)`.

| Position | Metric | y-axis | Scale |
|---|---|---|---|
| Top-left | Net Profit | `net_profit` (€) | linear |
| Top-right | Mean Step Time | `mean_step_ms` (ms) | log |
| Bottom-left | Served Customers | `served_customers` | linear |
| Bottom-right | Mean SoC Fulfillment | `mean_soc_fulfillment` | linear |

Each panel:
- x: horizon value, same ticks as Figure 1
- 95% CI: fill bands for linear panels, `ax.errorbar` for log panel
- One line per `ports` value, same colorblind palette

Legend: single legend in top-right panel (or outside if it overlaps). Panels whose metric column is absent from df call `ax.set_visible(False)` and are skipped.

x-axis label: "Horizon (steps)" on both bottom panels only (`sharex` not used — independent axes but same data range).

---

## Pipeline integration

In `__main__.py`, add after the existing `plot_compute_vs_horizon` call:

```python
plots.plot_horizon_comparison(df, output_dir)
plots.plot_horizon_full(df, output_dir)
```

Both functions guard against empty MILP subsets (same pattern as existing functions) and save nothing meaningful if there is no MILP data.

---

## Helper: `_horizon_agg(df, metric)`

Private helper shared by both functions:

```python
def _horizon_agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate metric by (horizon, ports) for MILP-only rows."""
    milp = df[df["horizon"].notna()].copy()
    agg = (
        milp.groupby(["horizon", "ports"])[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
    return agg
```

Avoids duplicating the filter + groupby + CI calculation in both functions.
