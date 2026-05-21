# Design: EV Benchmark Results Analysis Pipeline

**Date:** 2026-05-21
**Scope:** `src/evopt/analysis/` — new package; minimal additions to `run_chargax_benchmark.py` and `pyproject.toml`

---

## 1. Goals

Transform EV charging benchmark console output into a professional research-analysis pipeline suitable for a bachelor thesis, reproducible experimentation, and publication-quality figures. The pipeline reads a structured CSV/JSON export and produces:

- Summary tables (CSV + LaTeX)
- Publication-quality charts (PNG, 300 dpi)
- Statistical significance tests
- Correlation analysis
- Robustness metrics
- Markdown and self-contained HTML reports

---

## 2. Architecture

### Two-phase design

**Phase 1 — Export** (modification to existing benchmark script)

`run_chargax_benchmark.py` gains a `--save PATH` flag. When provided, it stamps each per-seed row with config columns (`experiment_id`, `ports`, `tariff`, `bess_enabled`, `v2g_enabled`, `solver`), concatenates all port configs into one combined DataFrame, writes it as CSV, and saves a `metadata.json` sidecar.

**Phase 2 — Analysis** (new `evopt.analysis` package)

Reads the saved CSV/JSON and produces all outputs into `--output DIR`.

```
run_chargax_benchmark --save results/benchmark.csv
        │
        ▼
  results/benchmark.csv  +  results/metadata.json
        │
        ▼
  python -m evopt.analysis --input results/benchmark.csv --output results/figures
        │
   ┌────┴────┬──────────┬─────────────┬──────────┐
loader   tables     plots      correlation   report
   │         │           │              │          │
clean   CSV+TEX    PNG 300dpi   corr.csv   .md + .html
derive  tables     charts       corr.md    benchmark_report
```

### Package structure

```
src/evopt/analysis/
    __init__.py
    utils.py        # shared: pub styling, save_figure, CI helper,
                    # horizon extraction, experiment ID, metadata helper
    loader.py       # load_results, clean_results, add_derived_metrics
    tables.py       # all summary tables → .csv + .tex
    plots.py        # all charts → .png at 300 dpi
    correlation.py  # Pearson matrix, highlight, CSV + MD summary
    stats.py        # paired t-test, Wilcoxon, Cohen's d, significance tables
    report.py       # benchmark_report.md + benchmark_report.html
    __main__.py     # CLI only — orchestrates the pipeline
```

---

## 3. Data Contracts

### Stage 1 — Raw export (written by benchmark, read by loader)

Columns in `benchmark.csv`:

| Column | Type | Description |
|---|---|---|
| `experiment_id` | str | e.g. `"dynamic_1.3_p3_6_12_h1_3_6_12"` |
| `controller` | str | e.g. `"milp_h6"`, `"equal_share"` |
| `seed` | int | 0 .. n_seeds−1 |
| `ports` | int | e.g. 3, 6, 12 |
| `tariff` | str | e.g. `"dynamic:1.3"` or `"0.75"` |
| `bess_enabled` | bool | |
| `v2g_enabled` | bool | |
| `solver` | str | e.g. `"highs"` |
| `net_profit` | float | € |
| `total_revenue` | float | € |
| `total_cost` | float | € |
| `served_customers` | int | |
| `rejected_customers` | int | |
| `mean_soc_fulfillment` | float | 0..1 |
| `gap_to_best` | float | ≤ 0 (overwritten by `clean_results`) |
| `total_compute_s` | float | seconds |
| `mean_step_ms` | float | milliseconds |

### Stage 2 — After `add_derived_metrics` (per-seed rows)

Two columns added:

| Column | Type | Description |
|---|---|---|
| `horizon` | float | integer from `milp_hN`; `NaN` for baselines |
| `profit_per_compute_s` | float | `net_profit / total_compute_s`; `NaN` when `total_compute_s == 0` |

`gap_to_best` is **recomputed** in `clean_results()`:
```python
df["gap_to_best"] = (
    df["net_profit"]
    - df.groupby(["seed", "ports"])["net_profit"].transform("max")
)
```
This ensures the gap is always ≤ 0 and is meaningful within the same experimental conditions (same seed + ports).

### Stage 3 — Aggregation (tables)

Grouped by `(controller, ports)`. Each metric yields:

| Column | Description |
|---|---|
| `mean` | arithmetic mean over seeds |
| `std` | standard deviation |
| `min` | minimum across seeds |
| `max` | maximum across seeds |
| `median` | median across seeds |
| `n` | number of seeds contributing |
| `ci_lower` | `mean − 1.96 · std / √n` |
| `ci_upper` | `mean + 1.96 · std / √n` |

### Stage 4 — Statistical tests (stats.py output)

Long-form DataFrame from `compare_all_controllers`:

| Column | Description |
|---|---|
| `controller_a` | str |
| `controller_b` | str |
| `metric` | str |
| `n_pairs` | int — matched pairs after alignment |
| `mean_a` | float |
| `mean_b` | float |
| `mean_diff` | `mean_a − mean_b` |
| `p_ttest` | float — paired t-test p-value |
| `p_wilcoxon` | float — Wilcoxon signed-rank p-value |
| `cohen_d` | float — effect size |
| `is_significant` | bool — `p_ttest < 0.05` |

### `metadata.json` schema

```json
{
  "experiment_id":  "dynamic_1.3_p3_6_12_h1_3_6_12",
  "timestamp":      "2026-05-21T14:32:00",
  "git_commit":     "775b03e",
  "cli_command":    "python -m evopt.experiments.run_chargax_benchmark ...",
  "hostname":       "DESKTOP-XYZ",
  "python_version": "3.11.9",
  "ports":          [3, 6, 12],
  "horizons":       [1, 3, 6, 12],
  "n_seeds":        10,
  "tariff":         "dynamic:1.3",
  "bess_enabled":   true,
  "v2g_enabled":    true,
  "solver":         "highs"
}
```

---

## 4. Module Specifications

### `utils.py`

Shared utilities used across all modules. No module in the package imports from `loader`, `tables`, `plots`, etc. — only from `utils`.

Key functions:

```python
apply_pub_style() -> None
    # Sets rcParams: font size 12, no top/right spines, colorblind palette,
    # tight_layout, figure size 8×5 default

save_figure(fig: Figure, path: Path, dpi: int = 300) -> None
    # Saves and closes figure; creates parent dirs

extract_horizon(controller_name: str) -> float
    # regex: milp_h(\d+) → int; else NaN

format_experiment_id(ports, horizons, tariff, **kwargs) -> str
    # human-readable: "dynamic_1.3_p3_6_12_h1_3_6_12"
    # sanitizes dots/slashes to underscores

ci95(series: pd.Series) -> tuple[float, float]
    # returns (ci_lower, ci_upper) using 1.96·std/√n

colorblind_palette(n: int) -> list[str]
    # returns n colors from the Okabe-Ito colorblind-safe palette
    # (8 colors: black, orange, sky-blue, bluish-green, yellow, blue, vermillion, reddish-purple)
    # cycles if n > 8

save_metadata(config: dict, output_dir: Path) -> None
    # collects git hash, hostname, python version, timestamp
    # writes metadata.json to output_dir
```

---

### `loader.py`

```python
load_results(path: Path) -> pd.DataFrame
    # detects .csv or .json by extension; raises ValueError for unknown formats

clean_results(df: pd.DataFrame) -> pd.DataFrame
    # drops rows where all metric columns are NaN
    # coerces numeric columns
    # recomputes gap_to_best within (seed, ports) groups

add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame
    # adds: horizon (float), profit_per_compute_s (NaN-safe)
```

---

### `tables.py`

All functions share signature `(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame`.
Each function saves `table_<name>.csv` and `table_<name>.tex`.
The `.tex` file uses `DataFrame.to_latex()` with float formatting rounded per metric.

| Function | Output | Contents |
|---|---|---|
| `make_full_table` | `table_full` | mean±CI for all metrics, grouped by controller × ports |
| `make_cross_port_table` | `table_cross_port` | pivot: controllers as rows, ports as columns; shows `mean` only per cell (CI would make the table unreadably wide — full CI is in `table_full`) |
| `make_best_controller_table` | `table_best` | best controller per port by mean net_profit |
| `make_compute_table` | `table_compute` | total_compute_s and mean_step_ms by controller |
| `make_efficiency_table` | `table_efficiency` | profit_per_compute_s with CI by controller × ports |
| `make_tradeoff_table` | `table_tradeoff` | net_profit, mean_step_ms, served, rejected side by side |
| `make_robustness_table` | `table_robustness` | coefficient of variation, IQR, worst/best case per controller |

**Robustness metrics** computed per `(controller, ports)` group:
- `cv` = std / mean (coefficient of variation)
- `iqr` = Q75 − Q25
- `worst_case` = min net_profit across seeds
- `best_case` = max net_profit across seeds

---

### `plots.py`

All functions share signature `(df: pd.DataFrame, output_dir: Path) -> None`.
All call `apply_pub_style()` before plotting and `save_figure()` after.

**Profit and performance charts:**

| Function | File | Description |
|---|---|---|
| `plot_profit_by_controller` | `profit_by_controller.png` | Grouped bar chart, x=controller, hue=ports, error bars=95% CI |
| `plot_profit_vs_horizon` | `profit_vs_horizon.png` | Line chart, MILP only, x=horizon, error band=95% CI |
| `plot_compute_vs_horizon` | `compute_vs_horizon.png` | Line chart, mean_step_ms vs horizon, MILP only, error bars=95% CI |

**Distribution / variance charts (per-seed data):**

| Function | File | Description |
|---|---|---|
| `plot_profit_boxplot` | `profit_boxplot.png` | Boxplot per controller, hue=ports |
| `plot_profit_violin` | `profit_violin.png` | Violin plot per controller, hue=ports |
| `plot_compute_boxplot` | `compute_boxplot.png` | Boxplot of total_compute_s per controller |

**Scatter / trade-off charts:**

| Function | File | Description |
|---|---|---|
| `plot_tradeoff_scatter` | `tradeoff_scatter.png` | 3 subplots: net_profit/mean_step_ms, served/rejected, soc/profit |
| `plot_pareto_frontier` | `pareto_frontier.png` | net_profit vs total_compute_s; Pareto-efficient controllers highlighted and annotated |

**Heatmaps:**

| Function | File | Description |
|---|---|---|
| `plot_correlation_heatmap` | `correlation_heatmap.png` | Seaborn heatmap of Pearson correlation matrix |
| `plot_profit_heatmap` | `profit_heatmap.png` | Mean net_profit by controller (rows) × ports (cols) |

**Scaling analysis:**

| Function | File | Description |
|---|---|---|
| `plot_scaling_compute` | `scaling_compute.png` | total_compute_s vs ports, one line per controller |
| `plot_scaling_profit` | `scaling_profit.png` | net_profit vs ports, one line per controller |
| `plot_scaling_served_customers` | `scaling_served_customers.png` | served_customers vs ports, one line per controller |

**Optional:**

| Function | File | Description |
|---|---|---|
| `plot_radar` | `radar.png` | Spider chart, normalized metrics per controller; not in default report |

---

### `correlation.py`

```python
compute_correlation(df: pd.DataFrame) -> pd.DataFrame
    # Pearson matrix on all numeric columns; excludes seed, ports

save_correlation_csv(corr: pd.DataFrame, output_dir: Path) -> None
    # writes correlation_matrix.csv

summarise_correlations(corr: pd.DataFrame) -> str
    # finds top 5 strongest positive and top 5 strongest negative pairs
    # returns formatted Markdown text
```

---

### `stats.py`

```python
compare_controllers(
    df: pd.DataFrame,
    metric: str,
    controller_a: str,
    controller_b: str,
    match_on: list[str] = ["seed", "ports"],
) -> dict
    # inner join on match_on to produce matched pairs
    # emits warning if n_pairs < n_seeds (some seeds unmatched)
    # returns: n_pairs, mean_a, mean_b, mean_diff,
    #          p_ttest, p_wilcoxon, cohen_d, is_significant

compare_all_controllers(
    df: pd.DataFrame,
    metric: str,
    match_on: list[str] = ["seed", "ports"],
) -> pd.DataFrame
    # pairwise comparison for all controller pairs
    # returns long-form DataFrame (see Stage 4 data contract)

save_significance_csv(results: pd.DataFrame, output_dir: Path) -> None
    # writes table_significance.csv + table_significance.tex

save_significance_summary(results: pd.DataFrame, output_dir: Path) -> None
    # writes significance_summary.md with human-readable findings
```

Uses `scipy.stats.ttest_rel` for paired t-test and `scipy.stats.wilcoxon` for signed-rank test.
Cohen's d computed as `mean_diff / pooled_std`.

---

### `report.py`

```python
write_markdown_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    corr_summary: str,
    sig_summary: str,
    output_dir: Path,
    exp_id: str,
) -> None
```

Report sections in `benchmark_report.md`:
1. Experiment Configuration (from metadata columns in df)
2. Key Findings (auto-generated narrative sentences, e.g.: "The best-performing controller by net_profit is `milp_h12` with mean €X.XX (95% CI [Y, Z]). The lowest-compute controller is `equal_share` at X ms/step. The highest efficiency (net_profit per compute second) is achieved by `milp_h1`. The strongest positive correlation is between `served_customers` and `net_profit` (r = X.XX).")
3. Best-Performing Controller (table_best embedded)
4. Full Results Table (table_full embedded)
5. Trade-off Analysis (table_tradeoff embedded)
6. Scalability Analysis (scaling chart links + narrative)
7. Statistical Significance Analysis (significance_summary embedded)
8. Correlation Analysis (corr_summary embedded)
9. Robustness Analysis (table_robustness embedded)

```python
write_html_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    exp_id: str,
) -> None
```

Same structure as Markdown report. All PNG charts embedded as base64 inline images. Tables rendered with `DataFrame.to_html(classes="table")` and minimal inline CSS. Output is a single self-contained `.html` file.

---

### `__main__.py`

CLI entry point only. No analysis logic lives here.

```
python -m evopt.analysis \
    --input  results/benchmark.csv \
    --output results/figures \
    [--no-radar]
```

Orchestration order:
1. `loader.load_results` → `clean_results` → `add_derived_metrics`
2. `tables.*` (all table functions)
3. `plots.*` (all plot functions; radar only if `--no-radar` not set)
4. `correlation.*`
5. `stats.compare_all_controllers` on `net_profit`
6. `report.write_markdown_report` + `write_html_report`
7. Print summary: list of files written

---

## 5. Changes to Existing Files

### `run_chargax_benchmark.py`

Additions only — no existing logic removed:

1. `--save PATH` argument added to `argparse` block
2. In `main()`: each per-port raw DataFrame gets `ports`, `experiment_id`, `tariff`, `bess_enabled`, `v2g_enabled`, `solver` columns stamped before being appended to `per_port_raw_dfs`
3. After the port loop: if `--save` provided, concatenate all raw DataFrames and write CSV; call `utils.save_metadata(config, save_path.parent)`
4. `gap_to_best` is **not** recomputed here — `loader.clean_results()` owns that correction

### `pyproject.toml`

Three new runtime dependencies:

```toml
"matplotlib>=3.7",
"seaborn>=0.12",
"scipy>=1.10",
```

### `src/evopt/benchmarking/storage.py`

No changes. The gap_to_best correction is owned by the analysis layer.

---

## 6. Testing

Test file: `tests/test_results_display.py`

**Unit tests (no I/O):**
- `test_extract_horizon_milp` — `milp_h6` → 6
- `test_extract_horizon_milp_double_digit` — `milp_h12` → 12
- `test_extract_horizon_baseline` — `equal_share` → NaN
- `test_gap_to_best_always_nonpositive` — all values ≤ 0
- `test_gap_to_best_within_seed_ports` — best controller in group has gap = 0
- `test_profit_per_compute_zero` — `total_compute_s == 0` → NaN, not inf
- `test_ci95_values` — known inputs produce expected CI bounds
- `test_derived_metrics_columns_exist` — horizon and profit_per_compute_s present
- `test_robustness_metrics` — cv, iqr, worst/best case computed correctly

**Statistical tests:**
- `test_compare_controllers_paired_alignment` — unmatched seeds are dropped before test
- `test_compare_controllers_identical` — comparing controller with itself → p=1.0, cohen_d=0
- `test_compare_all_controllers_shape` — output has correct number of rows

**Table tests (tmpdir):**
- `test_make_full_table_produces_csv` — file exists, has expected columns
- `test_make_full_table_produces_tex` — `.tex` file exists and is non-empty
- `test_make_robustness_table` — cv and iqr columns present

**Plotting tests (tmpdir, no display):**
All plotting functions called with a small 3-controller × 2-port × 5-seed sample DataFrame using `matplotlib.use("Agg")`. Assert output PNG file exists and is non-empty. Covered:
`plot_profit_by_controller`, `plot_profit_vs_horizon`, `plot_compute_vs_horizon`,
`plot_tradeoff_scatter`, `plot_correlation_heatmap`, `plot_profit_heatmap`,
`plot_profit_boxplot`, `plot_profit_violin`, `plot_compute_boxplot`,
`plot_scaling_compute`, `plot_scaling_profit`, `plot_scaling_served_customers`,
`plot_pareto_frontier`

**Metadata test:**
- `test_save_metadata_contains_keys` — output JSON has all required keys

---

## 7. CLI Usage Summary

```bash
# Step 1: run benchmark and save raw data
python -m evopt.experiments.run_chargax_benchmark \
    --seeds 10 --horizons 1 3 6 12 \
    --ports 3 6 12 --tariff dynamic:1.3 \
    --allow-discharging --allow-bess-discharging \
    --save results/benchmark.csv

# Step 2: generate all analysis outputs
python -m evopt.analysis \
    --input  results/benchmark.csv \
    --output results/figures

# Optional: skip radar chart
python -m evopt.analysis \
    --input  results/benchmark.csv \
    --output results/figures \
    --no-radar
```

Outputs written to `results/figures/`:
- `table_full.csv/.tex`, `table_cross_port.csv/.tex`, `table_best.csv/.tex`
- `table_compute.csv/.tex`, `table_efficiency.csv/.tex`, `table_tradeoff.csv/.tex`
- `table_robustness.csv/.tex`, `table_significance.csv/.tex`
- `correlation_matrix.csv`, `significance_summary.md`
- All PNG charts (300 dpi)
- `benchmark_report.md`
- `benchmark_report.html`
