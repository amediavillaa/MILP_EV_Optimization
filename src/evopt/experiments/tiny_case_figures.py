"""Publication figures and tables for the tiny cost case scenario."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.environ import value

from evopt.analysis.utils import (
    apply_pub_style,
    colorblind_palette,
    save_figure,
)
from evopt.experiments.run_tiny_cost_case import (
    _DATA,
    _LP_CAR_LABELS,
    _REJECTED,
)
from evopt.optimization.model import build_ev_lp_model
from evopt.optimization.solver import solve


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


# Five real-world cars in display order (Car 4 is rejected)
_SCENARIO_ROWS = [
    # (car_id, port, arrival, departure, s_init, s_target, status)
    ("Car 1", "1", "t1", "t4",  5,  20, "Served"),
    ("Car 2", "2", "t1", "t8", 10,  25, "Served"),
    ("Car 3", "3", "t1", "t8",  8,  22, "Served"),
    ("Car 4", "—", "t1", "—",  "—", "—", "Rejected"),
    ("Car 5", "1", "t5", "t8",  3,  15, "Served"),
]

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
    df = pd.DataFrame(_SCENARIO_ROWS, columns=_SCENARIO_COLS)
    df.to_csv(out / "table_scenario.csv", index=False)
    (out / "table_scenario.tex").write_text(_to_booktabs(df), encoding="utf-8")


def write_table_solution(res: TinyResults, out: Path) -> None:
    """Save table_solution.csv and table_solution.tex to *out*.

    Writes two sections:
      1. Summary metrics (Revenue, Grid cost, Net profit).
      2. Charging schedule (kW per port-car pair per timestep).
    """
    out = Path(out)
    T = res.T

    # --- section 1: summary ---
    summary = pd.DataFrame([
        {"Metric": "Revenue (€)",     "Value": f"{res.revenue:.2f}"},
        {"Metric": "Grid cost (€)",   "Value": f"{res.cost:.2f}"},
        {"Metric": "Net profit (€)",  "Value": f"{res.profit:.2f}"},
    ])

    # --- section 2: charging schedule ---
    port_labels = {
        1: "Port 1 / Car 1 (t1–t4)",
        2: "Port 2 / Car 2",
        3: "Port 3 / Car 3",
    }

    rows = []
    for j in [1, 2, 3]:
        row = {"Port / Car": port_labels[j]}
        for t in range(1, T + 1):
            row[f"t{t}"] = f"{res.power[j, t]:.2f}"
        rows.append(row)
    total_row = {"Port / Car": "Total (kW)"}
    for t in range(1, T + 1):
        total_row[f"t{t}"] = f"{sum(res.power[j, t] for j in [1, 2, 3]):.2f}"
    rows.append(total_row)
    schedule = pd.DataFrame(rows)

    # CSV: two sections separated by blank line
    csv_lines = []
    csv_lines.append("# Summary metrics")
    csv_lines.append(summary.to_csv(index=False).strip())
    csv_lines.append("")
    csv_lines.append("# Charging schedule (kW)")
    csv_lines.append(schedule.to_csv(index=False).strip())
    (out / "table_solution.csv").write_text("\n".join(csv_lines), encoding="utf-8")

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

        # Dotted target line spanning the dwell window
        ax.hlines(s_tgt, arr - 1, dep, colors=color, linestyles=":",
                  linewidth=1, alpha=0.55)

        # Star at first timestep target is reached
        for t in range(arr, dep + 1):
            if res.soc[lp_i, t] >= s_tgt - 1e-3:
                ax.plot(t, res.soc[lp_i, t], "*", color=color, markersize=10)
                break

    # Vertical boundary between Car 1 departure and Car 5 arrival
    ax.axvline(4.5, color="#888888", linestyle="--", linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.text(4.6, ymax * 0.97, "Car 1 dep\nCar 5 arr",
            fontsize=8, color="#666666", va="top")

    ax.set_xlabel("Timestep (h)")
    ax.set_ylabel("State of Charge (kWh)")
    ax.set_xticks(range(0, res.T + 1))
    ax.set_xticklabels(["t0"] + [f"t{t}" for t in range(1, res.T + 1)])
    ax.legend(loc="upper left", fontsize=9)
    save_figure(fig, Path(out) / "soc_trajectories.pdf")
