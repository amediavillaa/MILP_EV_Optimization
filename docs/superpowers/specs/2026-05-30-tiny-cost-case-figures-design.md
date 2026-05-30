# Design: Tiny Cost Case Figures

**Date:** 2026-05-30
**Status:** Approved

## Goal

Generate publication-quality figures and tables from `run_tiny_cost_case` for inclusion in an academic paper (single-column, ~6.5 in wide). All figures are PDF (vector). Tables are exported as LaTeX (`.tex`) and CSV.

## Scenario recap

3 ports, 8 hourly timesteps, 5 real-world cars (4 served, 1 rejected).

| Real-world car | LP car | Port | Arrival | Departure | s_init (kWh) | s_target (kWh) | Status |
|---|---|---|---|---|---|---|---|
| Car 1 | 1 | 1 | t1 | t4 | 5 | 20 | Served (tight deadline) |
| Car 2 | 2 | 2 | t1 | t8 | 10 | 25 | Served |
| Car 3 | 3 | 3 | t1 | t8 | 8 | 22 | Served |
| Car 4 | — | — | t1 | — | — | — | Rejected (no free port) |
| Car 5 | 4 | 1 | t5 | t8 | 3 | 15 | Served (late arrival, reuses port 1) |

Buy prices (€/kWh): t1–t2 = 0.20, t3–t4 = 0.18, t5–t6 = 0.15, t7–t8 = 0.12. Grid cap = 20 kW.

**Known solution:** Revenue = 54 €, Grid cost = 26 €, Net profit = 28 €. Grid cap is active (exactly 20 kW total) at every timestep.

## Implementation

**New file:** `src/evopt/experiments/tiny_case_figures.py`

- Imports `_DATA` and `_LP_CAR_LABELS` from `run_tiny_cost_case` (no data duplication)
- Solves the model once, extracts all values, then generates all figures and tables
- Entry point: `python -m evopt.experiments.tiny_case_figures`
- **Output directory:** `results/tiny_cost_case/` (created if absent)

## Figures

All figures: matplotlib, `apply_pub_style()` from `evopt.analysis.utils`, width = 6.5 in, saved as PDF.

### Figure 1 — `soc_trajectories.pdf`

Line chart. x = timestep (1–8), y = SoC (kWh).

- One line per LP car: Car 1 (blue, solid, active t0–t4), Car 2 (green, solid, t0–t8), Car 3 (red, solid, t0–t8), Car 5 (purple, dashed, t5–t8)
- x = 0 (arrival baseline, value = s_init) through t8; Car 5 starts at x = 5 (its s_init)
- Car lines are NaN-masked outside their dwell window (no line drawn when car is absent)
- Horizontal dotted line per car at its `s_target`
- ★ marker at the first timestep where `soc >= s_target - 1e-3`
- Vertical dashed grey line at x = 4.5 labelled "Car 1 dep / Car 5 arr"
- Legend inside upper-left; axis labels: "Timestep (h)" / "State of Charge (kWh)"

### Figure 2 — `charging_schedule.pdf`

Stacked bar chart. x = timestep, y = 0–25 kW.

- Four stack segments per timestep:
  - Port 1 / Car 1 (blue) — active t1–t4, zero t5–t8
  - Port 1 / Car 5 (purple) — zero t1–t4, active t5–t8
  - Port 2 / Car 2 (green) — all timesteps
  - Port 3 / Car 3 (red) — all timesteps
- Red dashed horizontal line at 20 kW labelled "Grid cap (20 kW)"; bars touch it at every step
- Vertical dashed grey line at x = 4.5 (Car 1 / Car 5 boundary)
- y-axis upper limit = 25 kW to leave visible gap above cap line
- Axis labels: "Timestep (h)" / "Charging Power (kW)"

### Figure 3 — `tariff_overlay.pdf`

Two-panel figure sharing the x-axis (gridspec, height ratio 1:2).

- **Top panel:** buy-price step function (orange), y = €/kWh, no x tick labels
- **Bottom panel:** same stacked bars as Figure 2 (same colours, same cap line, same boundary line)
- Single shared x-axis label "Timestep (h)" below the bottom panel

### Figure 4 — `scenario_gantt.pdf`

Horizontal Gantt chart. x = timestep (0.5–8.5), y = {Port 1, Port 2, Port 3, Rejected}.

- `barh` bars, height = 0.5:
  - Car 1: Port 1, t1–t4 (blue)
  - Car 5: Port 1, t5–t8 (purple)
  - Car 2: Port 2, t1–t8 (green)
  - Car 3: Port 3, t1–t8 (red)
  - Car 4: Rejected row, t1 only (grey, hatched, labelled "rejected")
- Car label printed inside or beside each bar
- x ticks at 1–8, labelled t1–t8
- No y-axis spine; minimal grid

## Tables

### `table_scenario.tex` / `table_scenario.csv`

One row per real-world car (5 rows including Car 4). Columns:

| Car | Port | Arrival | Departure | $s_0$ (kWh) | $s^*$ (kWh) | Status |
|---|---|---|---|---|---|---|

LaTeX: booktabs style (`\toprule`, `\midrule`, `\bottomrule`), column header with math mode for $s_0$ and $s^*$.

### `table_solution.tex` / `table_solution.csv`

Two sub-tables written to the same file:

1. **Summary metrics** (3 rows): Revenue, Grid cost, Net profit — values in €
2. **Charging schedule** (4 rows × 8 cols): kW per port (Port 1/Car 1, Port 1/Car 5, Port 2, Port 3) per timestep, plus a "Total" row

LaTeX: booktabs style, numeric columns right-aligned.

## Colour palette

Uses `colorblind_palette` from `evopt.analysis.utils` for the 4 car/port colours: blue (Car 1), purple (Car 5), green (Car 2), red (Car 3). Cap line: `#d62728`. Boundary line: `#888888`. Price line: `#e08a00`.

## File tree after running

```
results/tiny_cost_case/
├── soc_trajectories.pdf
├── charging_schedule.pdf
├── tariff_overlay.pdf
├── scenario_gantt.pdf
├── table_scenario.tex
├── table_scenario.csv
├── table_solution.tex
└── table_solution.csv
```
