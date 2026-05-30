# Tiny Cost Case Figures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `src/evopt/experiments/tiny_case_figures.py` — a standalone script that solves the tiny cost case, then saves 4 PDF figures and 4 table files (CSV + LaTeX) to `results/tiny_cost_case/`.

**Architecture:** One new module with a `TinyResults` dataclass, one `extract_results()` function, six generation functions (2 table, 4 figure), and a `main()` entry point. All figure functions accept a `TinyResults` instance and an output `Path`; the solver is called once in `main()`. Reuses `apply_pub_style`, `save_figure`, and `colorblind_palette` from `evopt.analysis.utils`.

**Tech Stack:** Python, Pyomo (solve), Matplotlib (figures), Pandas (tables), booktabs-style LaTeX output written manually.

**Spec:** `docs/superpowers/specs/2026-05-30-tiny-cost-case-figures-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/evopt/experiments/tiny_case_figures.py` | All generation logic + `main()` |
| Create | `tests/test_tiny_case_figures.py` | Tests for extraction, tables, figures, integration |

No other files touched.

---

## Task 1: Scaffold module and `extract_results`

**Files:**
- Create: `src/evopt/experiments/tiny_case_figures.py`
- Create: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Write failing tests for `extract_results`**

```python
# tests/test_tiny_case_figures.py
from __future__ import annotations
import pytest
from pathlib import Path

from evopt.experiments.run_tiny_cost_case import _DATA
from evopt.optimization.model import build_ev_lp_model
from evopt.optimization.solver import solve


@pytest.fixture(scope="module")
def res():
    from evopt.experiments.tiny_case_figures import extract_results
    m = build_ev_lp_model(_DATA)
    solve(m, solver="highs")
    return extract_results(m, _DATA)


def test_revenue(res):
    assert abs(res.revenue - 54.0) < 0.01

def test_cost(res):
    assert abs(res.cost - 26.0) < 0.01

def test_profit(res):
    assert abs(res.profit - 28.0) < 0.01

def test_grid_cap_never_exceeded(res):
    for t in range(1, 9):
        total = sum(res.power[j, t] for j in [1, 2, 3])
        assert total <= 20.0 + 1e-3, f"Grid cap exceeded at t={t}: {total:.3f} kW"

def test_car1_target_reached_by_departure(res):
    # Car 1 (LP i=1) must hit s_target=20 kWh by t=4
    assert res.soc[1, 4] >= 20.0 - 1e-3

def test_car5_target_reached(res):
    # Car 5 (LP i=4) must hit s_target=15 kWh by t=8
    assert res.soc[4, 8] >= 15.0 - 1e-3
```

- [ ] **Step 2: Run tests — expect ImportError / AttributeError**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: collection error — `tiny_case_figures` does not exist yet.

- [ ] **Step 3: Implement `TinyResults` and `extract_results`**

```python
# src/evopt/experiments/tiny_case_figures.py
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
```

- [ ] **Step 4: Run tests — all 6 should pass**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: scaffold tiny_case_figures with TinyResults and extract_results"
```

---

## Task 2: Scenario table (`table_scenario`)

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add failing tests**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_table_scenario_csv(tmp_path):
    from evopt.experiments.tiny_case_figures import write_table_scenario
    write_table_scenario(tmp_path)
    text = (tmp_path / "table_scenario.csv").read_text()
    assert "Car 4" in text
    assert "Rejected" in text
    assert "20" in text   # s_target for Car 1
    assert "Port" in text

def test_table_scenario_tex(tmp_path):
    from evopt.experiments.tiny_case_figures import write_table_scenario
    write_table_scenario(tmp_path)
    tex = (tmp_path / "table_scenario.tex").read_text()
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert "Rejected" in tex
    assert "Car 4" in tex
```

- [ ] **Step 2: Run new tests — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_table_scenario_csv tests/test_tiny_case_figures.py::test_table_scenario_tex -v
```

Expected: FAIL — `write_table_scenario` not defined.

- [ ] **Step 3: Implement `_to_booktabs` helper and `write_table_scenario`**

Add to `src/evopt/experiments/tiny_case_figures.py` (after the imports, before `main`):

```python
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
```

- [ ] **Step 4: Run all tests — expect 8 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add write_table_scenario (CSV + booktabs LaTeX)"
```

---

## Task 3: Solution table (`table_solution`)

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add failing tests**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_table_solution_csv_metrics(tmp_path, res):
    from evopt.experiments.tiny_case_figures import write_table_solution
    write_table_solution(res, tmp_path)
    text = (tmp_path / "table_solution.csv").read_text()
    assert "Revenue" in text
    assert "54" in text
    assert "26" in text
    assert "28" in text

def test_table_solution_tex(tmp_path, res):
    from evopt.experiments.tiny_case_figures import write_table_solution
    write_table_solution(res, tmp_path)
    tex = (tmp_path / "table_solution.tex").read_text()
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert "Revenue" in tex
```

- [ ] **Step 2: Run new tests — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_table_solution_csv_metrics tests/test_tiny_case_figures.py::test_table_solution_tex -v
```

Expected: FAIL — `write_table_solution` not defined.

- [ ] **Step 3: Implement `write_table_solution`**

Add to `src/evopt/experiments/tiny_case_figures.py`:

```python
def write_table_solution(res: TinyResults, out: Path) -> None:
    """Save table_solution.csv and table_solution.tex to *out*.

    Writes two sections:
      1. Summary metrics (Revenue, Grid cost, Net profit).
      2. Charging schedule (kW per port-car pair per timestep).
    """
    out = Path(out)
    T = res.T
    tsteps = [f"t{t}" for t in range(1, T + 1)]

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
    car5_label = "Port 1 / Car 5 (t5–t8)"

    rows = []
    for j in [1, 2, 3]:
        row = {"Port / Car": port_labels[j]}
        for t in range(1, T + 1):
            row[f"t{t}"] = f"{res.power[j, t]:.2f}"
        rows.append(row)
    # Car 5 uses port 1 from t5-t8; its power is already in res.power[1, t] for t≥5
    # But port 1 t1-t4 is Car 1 and port 1 t5-t8 is Car 5, both in res.power[1, t].
    # The schedule table already shows port 1 split above, so add a total row.
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

    # LaTeX: summary table only (schedule as-is is too wide for a single-column paper)
    (out / "table_solution.tex").write_text(_to_booktabs(summary), encoding="utf-8")
```

- [ ] **Step 4: Run all tests — expect 10 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add write_table_solution (CSV + LaTeX)"
```

---

## Task 4: Figure 1 — SoC trajectories

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_soc_trajectories_pdf(tmp_path, res):
    from evopt.experiments.tiny_case_figures import plot_soc_trajectories
    plot_soc_trajectories(res, tmp_path)
    pdf = tmp_path / "soc_trajectories.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert pdf.read_bytes()[:4] == b"%PDF"
```

- [ ] **Step 2: Run new test — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_soc_trajectories_pdf -v
```

Expected: FAIL — `plot_soc_trajectories` not defined.

- [ ] **Step 3: Implement `plot_soc_trajectories`**

Add to `src/evopt/experiments/tiny_case_figures.py`:

```python
# LP car index, display label, arrival t, departure t
_CAR_META = [
    (1, "Car 1 (port 1, dep t4)", 1, 4),
    (2, "Car 2 (port 2)",          1, 8),
    (3, "Car 3 (port 3)",          1, 8),
    (4, "Car 5 (port 1, arr t5)",  5, 8),
]


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
```

- [ ] **Step 4: Run all tests — expect 11 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add plot_soc_trajectories"
```

---

## Task 5: Figure 2 — Charging schedule

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_charging_schedule_pdf(tmp_path, res):
    from evopt.experiments.tiny_case_figures import plot_charging_schedule
    plot_charging_schedule(res, tmp_path)
    pdf = tmp_path / "charging_schedule.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert pdf.read_bytes()[:4] == b"%PDF"
```

- [ ] **Step 2: Run new test — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_charging_schedule_pdf -v
```

Expected: FAIL — `plot_charging_schedule` not defined.

- [ ] **Step 3: Implement `plot_charging_schedule`**

Add to `src/evopt/experiments/tiny_case_figures.py`:

```python
def plot_charging_schedule(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    T = res.T
    ts = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Four stack segments using module-level _C colour dict
    p1_car1 = np.array([res.power[1, t] if t <= 4 else 0.0 for t in range(1, T + 1)])
    p1_car5 = np.array([res.power[1, t] if t >= 5 else 0.0 for t in range(1, T + 1)])
    p2      = np.array([res.power[2, t] for t in range(1, T + 1)])
    p3      = np.array([res.power[3, t] for t in range(1, T + 1)])

    bar_kw = dict(width=0.6, align="center")
    ax.bar(ts, p1_car1, label="Port 1 / Car 1 (t1–t4)", color=_C[1], **bar_kw)
    ax.bar(ts, p1_car5, bottom=p1_car1,
           label="Port 1 / Car 5 (t5–t8)", color=_C[4], **bar_kw)
    ax.bar(ts, p2, bottom=p1_car1 + p1_car5,
           label="Port 2 / Car 2", color=_C[2], **bar_kw)
    ax.bar(ts, p3, bottom=p1_car1 + p1_car5 + p2,
           label="Port 3 / Car 3", color=_C[3], **bar_kw)

    # Grid cap line
    ax.axhline(20.0, color="#d62728", linestyle="--", linewidth=1.5,
               label="Grid cap (20 kW)")

    # Boundary between Car 1 and Car 5 phases
    ax.axvline(4.5, color="#888888", linestyle="--", linewidth=1)

    ax.set_ylim(0, 25)
    ax.set_xlabel("Timestep (h)")
    ax.set_ylabel("Charging Power (kW)")
    ax.set_xticks(range(1, T + 1))
    ax.set_xticklabels([f"t{t}" for t in range(1, T + 1)])
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, Path(out) / "charging_schedule.pdf")
```

- [ ] **Step 4: Run all tests — expect 12 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add plot_charging_schedule"
```

---

## Task 6: Figure 3 — Tariff overlay

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_tariff_overlay_pdf(tmp_path, res):
    from evopt.experiments.tiny_case_figures import plot_tariff_overlay
    plot_tariff_overlay(res, tmp_path)
    pdf = tmp_path / "tariff_overlay.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert pdf.read_bytes()[:4] == b"%PDF"
```

- [ ] **Step 2: Run new test — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_tariff_overlay_pdf -v
```

Expected: FAIL — `plot_tariff_overlay` not defined.

- [ ] **Step 3: Implement `plot_tariff_overlay`**

Add to `src/evopt/experiments/tiny_case_figures.py`:

```python
def plot_tariff_overlay(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    T = res.T
    ts = np.arange(1, T + 1)
    data = res.data

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

    # --- bottom panel: same stacked bars as plot_charging_schedule, using _C ---
    p1_car1 = np.array([res.power[1, t] if t <= 4 else 0.0 for t in range(1, T + 1)])
    p1_car5 = np.array([res.power[1, t] if t >= 5 else 0.0 for t in range(1, T + 1)])
    p2      = np.array([res.power[2, t] for t in range(1, T + 1)])
    p3      = np.array([res.power[3, t] for t in range(1, T + 1)])

    bar_kw = dict(width=0.6, align="center")
    ax_bot.bar(ts, p1_car1, label="Port 1 / Car 1 (t1–t4)", color=_C[1], **bar_kw)
    ax_bot.bar(ts, p1_car5, bottom=p1_car1,
               label="Port 1 / Car 5 (t5–t8)", color=_C[4], **bar_kw)
    ax_bot.bar(ts, p2, bottom=p1_car1 + p1_car5,
               label="Port 2 / Car 2", color=_C[2], **bar_kw)
    ax_bot.bar(ts, p3, bottom=p1_car1 + p1_car5 + p2,
               label="Port 3 / Car 3", color=_C[3], **bar_kw)
    ax_bot.axhline(20.0, color="#d62728", linestyle="--", linewidth=1.5,
                   label="Grid cap (20 kW)")
    ax_bot.axvline(4.5, color="#888888", linestyle="--", linewidth=1)
    ax_bot.set_ylim(0, 25)
    ax_bot.set_ylabel("Charging Power (kW)")
    ax_bot.set_xlabel("Timestep (h)")
    ax_bot.set_xticks(range(1, T + 1))
    ax_bot.set_xticklabels([f"t{t}" for t in range(1, T + 1)])
    ax_bot.legend(fontsize=8, loc="upper right")

    # shared boundary annotation on top panel
    ax_top.axvline(4.5, color="#888888", linestyle="--", linewidth=1)

    fig.tight_layout()
    save_figure(fig, Path(out) / "tariff_overlay.pdf")
```

- [ ] **Step 4: Run all tests — expect 13 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add plot_tariff_overlay (two-panel)"
```

---

## Task 7: Figure 4 — Scenario Gantt

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_scenario_gantt_pdf(tmp_path, res):
    from evopt.experiments.tiny_case_figures import plot_scenario_gantt
    plot_scenario_gantt(res, tmp_path)
    pdf = tmp_path / "scenario_gantt.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert pdf.read_bytes()[:4] == b"%PDF"
```

- [ ] **Step 2: Run new test — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_scenario_gantt_pdf -v
```

Expected: FAIL — `plot_scenario_gantt` not defined.

- [ ] **Step 3: Implement `plot_scenario_gantt`**

Add to `src/evopt/experiments/tiny_case_figures.py`:

```python
def plot_scenario_gantt(res: TinyResults, out: Path) -> None:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.5, 3))

    bar_h = 0.45
    # y positions for Port 1, Port 2, Port 3, Rejected row
    y_port = {1: 3, 2: 2, 3: 1}
    y_rej  = 0

    # Car 1: port 1, t1-t4  (uses _C[1] = blue)
    ax.barh(y_port[1], 4, left=0.5, height=bar_h, color=_C[1], label="Car 1")
    ax.text(0.5 + 4 / 2, y_port[1], "Car 1", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Car 5: port 1, t5-t8  (uses _C[4] = pink/purple)
    ax.barh(y_port[1], 4, left=4.5, height=bar_h, color=_C[4], label="Car 5")
    ax.text(4.5 + 4 / 2, y_port[1], "Car 5", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Car 2: port 2, t1-t8  (uses _C[2] = green)
    ax.barh(y_port[2], 8, left=0.5, height=bar_h, color=_C[2], label="Car 2")
    ax.text(0.5 + 8 / 2, y_port[2], "Car 2", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Car 3: port 3, t1-t8  (uses _C[3] = orange-red)
    ax.barh(y_port[3], 8, left=0.5, height=bar_h, color=_C[3], label="Car 3")
    ax.text(0.5 + 8 / 2, y_port[3], "Car 3", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Car 4: rejected — hatched bar at t1 on the Rejected row
    ax.barh(y_rej, 1, left=0.5, height=bar_h,
            color="none", edgecolor="#888888", linewidth=1.2,
            hatch="//", label="Car 4 (rejected)")
    ax.text(2.0, y_rej, "Car 4 — rejected (no free port at t1)",
            va="center", fontsize=8, color="#666666")

    ax.set_yticks([y_rej, y_port[3], y_port[2], y_port[1]])
    ax.set_yticklabels(["Rejected", "Port 3", "Port 2", "Port 1"])
    ax.set_xticks([t + 0.5 for t in range(0, 9)])
    ax.set_xticklabels([""] + [f"t{t}" for t in range(1, 9)] + [""])
    ax.set_xlim(0.2, 9.0)
    ax.set_xlabel("Timestep (h)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, Path(out) / "scenario_gantt.pdf")
```

- [ ] **Step 4: Run all tests — expect 14 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add plot_scenario_gantt"
```

---

## Task 8: `main()` entry point and integration test

**Files:**
- Modify: `src/evopt/experiments/tiny_case_figures.py`
- Modify: `tests/test_tiny_case_figures.py`

- [ ] **Step 1: Add integration test**

Add to `tests/test_tiny_case_figures.py`:

```python
def test_main_creates_all_outputs(tmp_path):
    from evopt.experiments.tiny_case_figures import main
    main(out_dir=tmp_path)
    expected = [
        "soc_trajectories.pdf",
        "charging_schedule.pdf",
        "tariff_overlay.pdf",
        "scenario_gantt.pdf",
        "table_scenario.csv",
        "table_scenario.tex",
        "table_solution.csv",
        "table_solution.tex",
    ]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing output: {fname}"
```

- [ ] **Step 2: Run new test — expect NameError**

```
pytest tests/test_tiny_case_figures.py::test_main_creates_all_outputs -v
```

Expected: FAIL — `main` not defined.

- [ ] **Step 3: Implement `main()` and `__main__` block**

Add to `src/evopt/experiments/tiny_case_figures.py`:

```python
def main(out_dir: Path = Path("results/tiny_cost_case")) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = build_ev_lp_model(_DATA)
    solve(m, solver="highs")
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
```

- [ ] **Step 4: Run full test suite — expect 15 passed**

```
pytest tests/test_tiny_case_figures.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Smoke-run the script end-to-end**

```
python -m evopt.experiments.tiny_case_figures
```

Expected output:
```
Figures and tables saved to results/tiny_cost_case/
```

Check the directory:
```
ls results/tiny_cost_case/
```

Expected 8 files:
```
charging_schedule.pdf  scenario_gantt.pdf  soc_trajectories.pdf  tariff_overlay.pdf
table_scenario.csv     table_scenario.tex  table_solution.csv    table_solution.tex
```

- [ ] **Step 6: Commit**

```bash
git add src/evopt/experiments/tiny_case_figures.py tests/test_tiny_case_figures.py
git commit -m "feat: add main() entry point for tiny_case_figures"
```
