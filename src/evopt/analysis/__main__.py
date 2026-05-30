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

    print(f"Loading {input_path} ...")
    df = add_derived_metrics(clean_results(load_results(input_path)))
    exp_id = df["experiment_id"].iloc[0] if "experiment_id" in df.columns else "benchmark"

    print("Generating tables ...")
    tbls = {
        "full":       tables.make_full_table(df, output_dir),
        "cross_port": tables.make_cross_port_table(df, output_dir),
        "best":       tables.make_best_controller_table(df, output_dir),
        "compute":    tables.make_compute_table(df, output_dir),
        "efficiency": tables.make_efficiency_table(df, output_dir),
        "tradeoff":   tables.make_tradeoff_table(df, output_dir),
        "robustness": tables.make_robustness_table(df, output_dir),
    }

    print("Generating charts ...")
    plots.plot_profit_by_controller(df, output_dir)
    plots.plot_profit_vs_horizon(df, output_dir)
    plots.plot_compute_vs_horizon(df, output_dir)
    plots.plot_horizon_comparison(df, output_dir)
    plots.plot_horizon_full(df, output_dir)
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

    print("Computing correlation ...")
    corr = correlation.compute_correlation(df)
    correlation.save_correlation_csv(corr, output_dir)
    corr_summary = correlation.summarise_correlations(corr)

    print("Running significance tests ...")
    sig_results = stats.compare_all_controllers(df, "net_profit")
    stats.save_significance_csv(sig_results, output_dir)
    stats.save_significance_summary(sig_results, output_dir)
    sig_summary = (output_dir / "significance_summary.md").read_text(encoding="utf-8")

    print("Writing reports ...")
    report.write_markdown_report(df, tbls, corr_summary, sig_summary, output_dir, exp_id)
    report.write_html_report(df, tbls, output_dir, exp_id)

    # Summary of outputs
    files = sorted(output_dir.iterdir())
    print(f"\nOutputs written to {output_dir}/")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
