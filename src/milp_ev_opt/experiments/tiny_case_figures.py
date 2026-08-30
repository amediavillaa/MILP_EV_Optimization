"""Publication figures and tables for the tiny cost case scenario."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.environ import value
from pyomo.opt import SolverStatus, TerminationCondition

from milp_ev_opt.analysis.utils import (
    apply_pub_style,
    colorblind_palette,
    save_figure,
)
from milp_ev_opt.experiments.run_tiny_cost_case import _DATA
from milp_ev_opt.optimization.model import build_ev_lp_model
from milp_ev_opt.optimization.solver import solve


@dataclass
class TinyResults:
    T: int
    power: dict   # {(port_j, t): float kW}
    soc: dict     # {(lp_car_i, t): float kWh}
    revenue: float
    cost: float
    profit: float
    data: dict


# Okabe-Ito indices for each LP car (avoids black at index 0)
# colorblind_palette(8) → ["#000000","#E69F00","#56B4E9","#009E73",
#                           "#F0E442","#0072B2","#D55E00","#CC79A7"]
_PAL = colorblind_palette(8)
_C = {
    1: _PAL[5],   # Car 1 → blue        #0072B2
    2: _PAL[3],   # Car 2 → green       #009E73
    3: _PAL[6],   # Car 3 → orange-red  #D55E00
    4: _PAL[7],   # Car 5 (LP 4) → pink #CC79A7
}

# Real-world display name for each LP car index
_LP_REAL_NAME = {1: "Car 1", 2: "Car 2", 3: "Car 3", 4: "Car 5"}


def _build_scenario_rows() -> list[tuple]:
    """Build scenario table rows from _DATA so they stay in sync with the LP."""
    d = _DATA
    rows = []
    for lp_i in [1, 2, 3]:
        rows.append((
            _LP_REAL_NAME[lp_i],
            str(d["assignments"][lp_i]),
            f"t{d['arr'][lp_i]}",
            f"t{d['dep'][lp_i]}",
            int(d["s_init"][lp_i]),
            int(d["s_target"][lp_i]),
            "Served",
        ))
    # Car 4 was rejected before the LP was built; not present in _DATA.
    rows.append(("Car 4", "—", "t1", "—", "—", "—", "Rejected"))
    rows.append((
        _LP_REAL_NAME[4],
        str(d["assignments"][4]),
        f"t{d['arr'][4]}",
        f"t{d['dep'][4]}",
        int(d["s_init"][4]),
        int(d["s_target"][4]),
        "Served",
    ))
    return rows


_SCENARIO_ROWS = _build_scenario_rows()

_SCENARIO_COLS = ["Car", "Port", "Arrival", "Departure",
                  r"$s_0$ (kWh)", r"$s^*$ (kWh)", "Status"]


# LP car index, display label, arrival t, departure t
_CAR_META = [
    (1, "Car 1 (port 1, dep t4)", 1, 4),
    (2, "Car 2 (port 2)",          1, 8),
    (3, "Car 3 (port 3)",          1, 8),
    (4, "Car 5 (port 1, arr t5)",  5, 8),
]


def _to_booktabs(df: pd.DataFrame) -> str:
    """Return a booktabs LaTeX tabular string for *df*."""
    col_fmt = "l" + "r" * (len(df.columns) - 1)
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    lines = [
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_table_scenario(out: Path) -> None:
    """Save table_scenario.csv and table_scenario.tex to *out*."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_SCENARIO_ROWS, columns=_SCENARIO_COLS)
    df.to_csv(out / "table_scenario.csv", index=False)
    (out / "table_scenario.tex").write_text(_to_booktabs(df), encoding="utf-8")


def write_table_solution(res: TinyResults, out: Path) -> None:
    """Save table_solution.csv and table_solution.tex to *out*.

    Writes two sections:
      1. Summary metrics (Revenue, Grid cost, Net profit).
      2. Charging schedule (kW per port-car pair per timestep), with Port 1
         split into Car 1 and Car 5 rows to reflect the port-reuse scenario.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    T = res.T
    dep_car1 = res.data["dep"][1]   # LP car 1 departs after this step
    arr_car5 = res.data["arr"][4]   # LP car 4 (Car 5) arrives at this step

    # --- section 1: summary ---
    summary = pd.DataFrame([
        {"Metric": "Revenue (€)",     "Value": f"{res.revenue:.2f}"},
        {"Metric": "Grid cost (€)",   "Value": f"{res.cost:.2f}"},
        {"Metric": "Net profit (€)",  "Value": f"{res.profit:.2f}"},
    ])

    # --- section 2: charging schedule ---
    # Port 1 is split: Car 1 occupies t1..dep_car1, Car 5 occupies arr_car5..T
    rows = []

    row = {"Port / Car": f"Port 1 / Car 1 (t1–t{dep_car1})"}
    for t in range(1, T + 1):
        row[f"t{t}"] = f"{res.power[1, t] if t <= dep_car1 else 0.0:.2f}"
    rows.append(row)

    row = {"Port / Car": f"Port 1 / Car 5 (t{arr_car5}–t{T})"}
    for t in range(1, T + 1):
        row[f"t{t}"] = f"{res.power[1, t] if t >= arr_car5 else 0.0:.2f}"
    rows.append(row)

    for j, name in [(2, "Port 2 / Car 2"), (3, "Port 3 / Car 3")]:
        row = {"Port / Car": name}
        for t in range(1, T + 1):
            row[f"t{t}"] = f"{res.power[j, t]:.2f}"
        rows.append(row)

    total_row = {"Port / Car": "Total (kW)"}
    for t in range(1, T + 1):
        total_row[f"t{t}"] = f"{sum(res.power[j, t] for j in range(1, res.data['J'] + 1)):.2f}"
    rows.append(total_row)

    schedule = pd.DataFrame(rows)

    # CSV: two sections separated by blank line
    csv_lines = [
        "# Summary metrics",
        summary.to_csv(index=False, lineterminator='\n').strip(),
        "",
        "# Charging schedule (kW)",
        schedule.to_csv(index=False, lineterminator='\n').strip(),
    ]
    (out / "table_solution.csv").write_text(
        "\n".join(csv_lines), encoding="utf-8", newline='\n'
    )

    # LaTeX: summary table only (schedule is too wide for single-column paper)
    (out / "table_solution.tex").write_text(_to_booktabs(summary), encoding="utf-8")


def extract_results(m, data: dict) -> TinyResults:
    T = data["T"]
    J = data["J"]
    I = data["I"]
    power = {
        (j, t): value(m.I_ev[j, t]) * value(m.V[j]) / 1000.0
        for j in range(1, J + 1)
        for t in range(1, T + 1)
    }
    soc = {
        (i, t): value(m.soc_car[i, t])
        for i in range(1, I + 1)
        for t in range(1, T + 1)
    }
    revenue = sum(
        value(m.I_ev[j, t]) * value(m.V[j]) * (value(m.delta_t) / 1000.0)
        * value(m.p_sell[t]) * value(m.z[j, t])
        for j in m.J_ev for t in m.T
    )
    cost = sum(
        value(m.I_ev[j, t]) * value(m.V[j]) * (value(m.delta_t) / 1000.0)
        * value(m.p_buy[t])
        for j in m.J_ev for t in m.T
    )
    return TinyResults(
        T=T,
        power=power,
        soc=soc,
        revenue=revenue,
        cost=cost,
        profit=revenue - cost,
        data=data,
    )


def plot_soc_trajectories(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    data = res.data
    dep_car1 = data["dep"][1]

    for (lp_i, label, arr, dep) in _CAR_META:
        color = _C[lp_i]
        ls = "--" if lp_i == 4 else "-"
        s_init = data["s_init"][lp_i]
        s_tgt  = data["s_target"][lp_i]

        # Build (x, y) pairs: prepend initial SoC just before first active step
        xs = [arr - 1] + list(range(arr, dep + 1))
        ys = [s_init]  + [res.soc[lp_i, t] for t in range(arr, dep + 1)]

        ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                markersize=3, label=label)

        # Dotted target line spanning the dwell window only (not pre-arrival anchor)
        ax.hlines(s_tgt, arr, dep, colors=color, linestyles=":",
                  linewidth=1, alpha=0.55)

        # Star at first timestep target is reached
        for t in range(arr, dep + 1):
            if res.soc[lp_i, t] >= s_tgt - 1e-3:
                ax.plot(t, res.soc[lp_i, t], "*", color=color, markersize=10)
                break

    # Vertical boundary between Car 1 departure and Car 5 arrival
    boundary = dep_car1 + 0.5
    ax.axvline(boundary, color="#888888", linestyle="--", linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.text(boundary + 0.1, ymax * 0.97, "Car 1 dep\nCar 5 arr",
            fontsize=8, color="#666666", va="top")

    ax.set_xlabel("Timestep (h)")
    ax.set_ylabel("State of Charge (kWh)")
    ax.set_xticks(range(0, res.T + 1))
    ax.set_xticklabels(["t0"] + [f"t{t}" for t in range(1, res.T + 1)])
    ax.legend(loc="upper left", fontsize=9)
    save_figure(fig, Path(out) / "soc_trajectories.pdf")


def plot_charging_schedule(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    T = res.T
    ts = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))

    dep_car1 = res.data["dep"][1]
    arr_car5 = res.data["arr"][4]

    p1_car1 = np.array([res.power[1, t] if t <= dep_car1 else 0.0 for t in range(1, T + 1)])
    p1_car5 = np.array([res.power[1, t] if t >= arr_car5 else 0.0 for t in range(1, T + 1)])
    p2      = np.array([res.power[2, t] for t in range(1, T + 1)])
    p3      = np.array([res.power[3, t] for t in range(1, T + 1)])

    bar_kw = dict(width=0.6, align="center")
    ax.bar(ts, p1_car1, label=f"Port 1 / Car 1 (t1–t{dep_car1})", color=_C[1], **bar_kw)
    ax.bar(ts, p1_car5, bottom=p1_car1,
           label=f"Port 1 / Car 5 (t{arr_car5}–t{T})", color=_C[4], **bar_kw)
    ax.bar(ts, p2, bottom=p1_car1 + p1_car5,
           label="Port 2 / Car 2", color=_C[2], **bar_kw)
    ax.bar(ts, p3, bottom=p1_car1 + p1_car5 + p2,
           label="Port 3 / Car 3", color=_C[3], **bar_kw)

    # Grid cap line
    ax.axhline(20.0, color="#d62728", linestyle="--", linewidth=1.5,
               label="Grid cap (20 kW)")

    # Boundary between Car 1 and Car 5 phases
    ax.axvline(dep_car1 + 0.5, color="#888888", linestyle="--", linewidth=1)

    ax.set_ylim(0, 25)
    ax.set_xlabel("Timestep (h)")
    ax.set_ylabel("Charging Power (kW)")
    ax.set_xticks(range(1, T + 1))
    ax.set_xticklabels([f"t{t}" for t in range(1, T + 1)])
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, Path(out) / "charging_schedule.pdf")


def plot_tariff_overlay(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    T = res.T
    ts = np.arange(1, T + 1)
    data = res.data
    dep_car1 = data["dep"][1]
    arr_car5 = data["arr"][4]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 5),
        gridspec_kw={"height_ratios": [1, 2]},
        sharex=True,
    )

    # --- top panel: buy price step function ---
    prices = [data["p_buy"][t] for t in range(1, T + 1)]
    ax_top.step(ts, prices, where="mid", color="#e08a00", linewidth=2)
    ax_top.set_ylabel("Buy Price (€/kWh)")
    ax_top.set_ylim(0, max(prices) * 1.3)

    # --- bottom panel: same stacked bars as plot_charging_schedule ---
    p1_car1 = np.array([res.power[1, t] if t <= dep_car1 else 0.0 for t in range(1, T + 1)])
    p1_car5 = np.array([res.power[1, t] if t >= arr_car5 else 0.0 for t in range(1, T + 1)])
    p2      = np.array([res.power[2, t] for t in range(1, T + 1)])
    p3      = np.array([res.power[3, t] for t in range(1, T + 1)])

    bar_kw = dict(width=0.6, align="center")
    ax_bot.bar(ts, p1_car1, label=f"Port 1 / Car 1 (t1–t{dep_car1})", color=_C[1], **bar_kw)
    ax_bot.bar(ts, p1_car5, bottom=p1_car1,
               label=f"Port 1 / Car 5 (t{arr_car5}–t{T})", color=_C[4], **bar_kw)
    ax_bot.bar(ts, p2, bottom=p1_car1 + p1_car5,
               label="Port 2 / Car 2", color=_C[2], **bar_kw)
    ax_bot.bar(ts, p3, bottom=p1_car1 + p1_car5 + p2,
               label="Port 3 / Car 3", color=_C[3], **bar_kw)
    ax_bot.axhline(20.0, color="#d62728", linestyle="--", linewidth=1.5,
                   label="Grid cap (20 kW)")
    boundary = dep_car1 + 0.5
    ax_bot.axvline(boundary, color="#888888", linestyle="--", linewidth=1)
    ax_bot.set_ylim(0, 25)
    ax_bot.set_ylabel("Charging Power (kW)")
    ax_bot.set_xlabel("Timestep (h)")
    ax_bot.set_xticks(range(1, T + 1))
    ax_bot.set_xticklabels([f"t{t}" for t in range(1, T + 1)])
    ax_bot.legend(fontsize=8, loc="upper right")

    ax_top.axvline(boundary, color="#888888", linestyle="--", linewidth=1)

    fig.tight_layout()
    save_figure(fig, Path(out) / "tariff_overlay.pdf")


def plot_scenario_gantt(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.5, 3))

    bar_h = 0.45
    y_port = {1: 3, 2: 2, 3: 1}
    y_rej  = 0

    d = res.data

    def _left(arr: int) -> float:
        return arr - 0.5

    def _width(arr: int, dep: int) -> int:
        return dep - arr + 1

    # Car 1: port 1
    lp1_arr, lp1_dep = d["arr"][1], d["dep"][1]
    ax.barh(y_port[1], _width(lp1_arr, lp1_dep), left=_left(lp1_arr),
            height=bar_h, color=_C[1], label="Car 1")
    ax.text(_left(lp1_arr) + _width(lp1_arr, lp1_dep) / 2, y_port[1], "Car 1",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Car 5 (LP car 4): port 1, reuses after Car 1
    lp4_arr, lp4_dep = d["arr"][4], d["dep"][4]
    ax.barh(y_port[1], _width(lp4_arr, lp4_dep), left=_left(lp4_arr),
            height=bar_h, color=_C[4], label="Car 5")
    ax.text(_left(lp4_arr) + _width(lp4_arr, lp4_dep) / 2, y_port[1], "Car 5",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Car 2: port 2
    lp2_arr, lp2_dep = d["arr"][2], d["dep"][2]
    ax.barh(y_port[2], _width(lp2_arr, lp2_dep), left=_left(lp2_arr),
            height=bar_h, color=_C[2], label="Car 2")
    ax.text(_left(lp2_arr) + _width(lp2_arr, lp2_dep) / 2, y_port[2], "Car 2",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Car 3: port 3
    lp3_arr, lp3_dep = d["arr"][3], d["dep"][3]
    ax.barh(y_port[3], _width(lp3_arr, lp3_dep), left=_left(lp3_arr),
            height=bar_h, color=_C[3], label="Car 3")
    ax.text(_left(lp3_arr) + _width(lp3_arr, lp3_dep) / 2, y_port[3], "Car 3",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Car 4: rejected — shown as hatched bar at its arrival timestep
    ax.barh(y_rej, 1, left=0.5, height=bar_h,
            color="none", edgecolor="#888888", linewidth=1.2,
            hatch="//", label="Car 4 (rejected)")
    ax.text(2.0, y_rej, "Car 4 — rejected (no free port at t1)",
            va="center", fontsize=8, color="#666666")

    ax.set_yticks([y_rej, y_port[3], y_port[2], y_port[1]])
    ax.set_yticklabels(["Rejected", "Port 3", "Port 2", "Port 1"])
    # Centre-of-slot ticks: each timestep t occupies x=[t-0.5, t+0.5], label at x=t
    ax.set_xticks(range(1, res.T + 1))
    ax.set_xticklabels([f"t{t}" for t in range(1, res.T + 1)])
    ax.set_xlim(0.3, res.T + 0.7)
    ax.set_xlabel("Timestep (h)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, Path(out) / "scenario_gantt.pdf")


def main(out_dir: Path = Path("results/tiny_cost_case")) -> None:
    """Generate all publication figures and tables for the tiny cost case.

    Parameters
    ----------
    out_dir : Path
        Output directory for figures and tables. Default: results/tiny_cost_case
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = build_ev_lp_model(_DATA)
    result = solve(m, solver="highs")
    if (result.solver.status != SolverStatus.ok
            or result.solver.termination_condition != TerminationCondition.optimal):
        raise RuntimeError(
            f"HiGHS did not find an optimal solution "
            f"(status={result.solver.status}, "
            f"termination={result.solver.termination_condition})"
        )
    res = extract_results(m, _DATA)

    plot_soc_trajectories(res, out_dir)
    plot_charging_schedule(res, out_dir)
    plot_tariff_overlay(res, out_dir)
    plot_scenario_gantt(res, out_dir)
    write_table_scenario(out_dir)
    write_table_solution(res, out_dir)

    print(f"Figures and tables saved to {out_dir}/")


if __name__ == "__main__":
    main()
