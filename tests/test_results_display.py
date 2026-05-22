# tests/test_results_display.py
from __future__ import annotations
import math, warnings
import subprocess, sys
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
    with pytest.warns(UserWarning, match="distinct experiment_id"):
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


# ── tables ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
from evopt.analysis.tables import (
    make_full_table, make_cross_port_table, make_best_controller_table,
    make_compute_table, make_efficiency_table, make_tradeoff_table,
    make_robustness_table,
)

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
    assert len(result) == _ready_df()["ports"].nunique()

def test_make_robustness_table(tmp_path):
    result = make_robustness_table(_ready_df(), tmp_path)
    assert (tmp_path / "table_robustness.csv").exists()
    col_str = str(result.columns.tolist())
    assert "cv" in col_str
    assert "iqr" in col_str
    assert "worst_case" in col_str
    assert "best_case" in col_str


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

# ── plots (distribution + scatter) ─────────────────────────────────────────
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

# ── plots (heatmaps + scaling) ──────────────────────────────────────────────
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

# ── plots (radar chart) ─────────────────────────────────────────────────────
from evopt.analysis.plots import plot_radar

def test_plot_radar(tmp_path):
    plot_radar(_plot_df(), tmp_path)
    _png_nonempty(tmp_path / "radar.png")

# ── correlation ─────────────────────────────────────────────────────────────
from evopt.analysis.correlation import (
    compute_correlation, save_correlation_csv, summarise_correlations,
)

def test_compute_correlation_shape():
    df = _ready_df()
    corr = compute_correlation(df)
    assert corr.shape[0] == corr.shape[1]          # square
    assert np.allclose(corr.values.diagonal(), 1.0)

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
    """cohen_d must be mean(diff)/std(diff), not mean_diff/pooled_std.

    A small noise term is added to the offset so that diff.std(ddof=1) > 0
    and expected_d is finite — otherwise the assertion is numerically
    unsatisfiable (inf - inf = nan).
    """
    rng = np.random.default_rng(42)
    rows = []
    for seed in range(20):
        base = rng.normal(10, 1)
        noise = rng.normal(0, 0.01)  # keeps diff variance non-zero
        rows.append({"controller": "A", "seed": seed, "ports": 3,
                     "experiment_id": "e", "net_profit": base})
        rows.append({"controller": "B", "seed": seed, "ports": 3,
                     "experiment_id": "e", "net_profit": base + 2.0 + noise})
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


# ── report ──────────────────────────────────────────────────────────────────
from evopt.analysis.report import write_markdown_report, write_html_report

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


# ── CLI ─────────────────────────────────────────────────────────────────────

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
