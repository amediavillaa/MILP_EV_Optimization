# src/evopt/analysis/plots.py
"""Publication-quality charts for EV benchmark results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evopt.analysis.utils import apply_pub_style, colorblind_palette, save_figure


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


def plot_profit_by_controller(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: mean net_profit by controller, grouped by ports.

    Error bars show 95% confidence intervals over seeds.
    """
    apply_pub_style()
    ports_vals = sorted(df["ports"].unique())
    controllers = list(df["controller"].unique())
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

    Error bands show 95% CI. Only MILP controllers (milp_hN) are included.
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
    ax.set_yscale("log")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Mean Step Time (ms)")
    ax.set_title("MILP Compute Time vs Horizon (95% CI)")
    ax.legend(title="Ports")
    save_figure(fig, Path(output_dir) / "compute_vs_horizon.png")


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


def plot_profit_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of per-seed net_profit per controller, coloured by ports."""
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ports_vals = sorted(df["ports"].unique())
    palette = {p: c for p, c in zip(ports_vals, colorblind_palette(len(ports_vals)))}
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
    ports_vals = sorted(df["ports"].unique())
    palette = {p: c for p, c in zip(ports_vals, colorblind_palette(len(ports_vals)))}
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df, x="controller", y="net_profit", hue="ports",
        palette=palette, ax=ax, inner="quart",
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
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df, x="controller", y="total_compute_s", ax=ax,
        color=colorblind_palette(1)[0],
        flierprops={"marker": "o", "markersize": 3},
    )
    ax.set_yscale("log")
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
        ("mean_step_ms",        "net_profit",           "Step Time (ms)",      "Net Profit (€)",       "Profit vs Step Time"),
        ("rejected_customers",  "served_customers",     "Rejected Customers",  "Served Customers",     "Served vs Rejected"),
        ("net_profit",          "mean_soc_fulfillment", "Net Profit (€)",      "Mean SoC Fulfillment", "SoC Fulfilment vs Profit"),
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
    """Net profit vs total_compute_s; Pareto-efficient controllers highlighted."""
    apply_pub_style()
    agg = df.groupby("controller")[["net_profit", "total_compute_s"]].mean().reset_index()

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


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Seaborn heatmap of Pearson correlation matrix across all numeric metrics."""
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
    """Heatmap: mean net_profit by controller (rows) x ports (cols)."""
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
    yscale: str = "linear",
) -> None:
    apply_pub_style()
    controllers = sorted(df["controller"].unique())
    palette = colorblind_palette(len(controllers))
    agg = df.groupby(["controller", "ports"])[metric].mean().reset_index()
    fig, ax = plt.subplots()
    for ctrl, color in zip(controllers, palette):
        sub = agg[agg["controller"] == ctrl].sort_values("ports")
        ax.plot(sub["ports"], sub[metric], marker="o", label=ctrl, color=color)
    ax.set_yscale(yscale)
    ax.set_xlabel("Number of Ports")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Controller", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    save_figure(fig, Path(output_dir) / filename)


def plot_scaling_compute(df: pd.DataFrame, output_dir: Path) -> None:
    """Total compute time (s) vs ports, one line per controller."""
    _plot_scaling(df, output_dir, "total_compute_s", "Total Compute Time (s)",
                  "scaling_compute.png", "Compute Time Scaling with Port Count",
                  yscale="log")


def plot_scaling_profit(df: pd.DataFrame, output_dir: Path) -> None:
    """Net profit (€) vs ports, one line per controller."""
    _plot_scaling(df, output_dir, "net_profit", "Net Profit (€)",
                  "scaling_profit.png", "Net Profit Scaling with Port Count")


def plot_scaling_served_customers(df: pd.DataFrame, output_dir: Path) -> None:
    """Served customers vs ports, one line per controller."""
    _plot_scaling(df, output_dir, "served_customers", "Served Customers",
                  "scaling_served_customers.png",
                  "Served Customers Scaling with Port Count")


def plot_radar(df: pd.DataFrame, output_dir: Path) -> None:
    """Optional spider chart: normalised metrics per controller.

    Not included in the default report. Metrics are min-max normalised so all
    axes share [0, 1]. gap_to_best and mean_step_ms are inverted (higher=better).
    """
    apply_pub_style()
    _RADAR_METRICS = [m for m in [
        "net_profit", "served_customers", "mean_soc_fulfillment",
        "gap_to_best", "mean_step_ms",
    ] if m in df.columns]

    agg = df.groupby("controller")[_RADAR_METRICS].mean()
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
