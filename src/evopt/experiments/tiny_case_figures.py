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
