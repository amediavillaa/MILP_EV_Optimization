# src/milp_ev_opt/analysis/report.py
"""Generate benchmark_report.md and benchmark_report.html."""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd


def _df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    try:
        return df.head(max_rows).to_markdown(floatfmt=".3f")
    except (ImportError, Exception):
        return df.head(max_rows).to_string()


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

    (output_dir / "benchmark_report.md").write_text("\n".join(sections), encoding="utf-8")


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

    (output_dir / "benchmark_report.html").write_text("\n".join(html_parts), encoding="utf-8")
