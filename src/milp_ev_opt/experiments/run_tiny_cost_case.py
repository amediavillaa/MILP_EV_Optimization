"""
run_tiny_cost_case.py — Small reference scenario for the EV charging LP
========================================================================
Setup:
  3 ports,  8 time steps (1 h each),  5 real-world cars
  4 cars served,  1 rejected (no free port on arrival)

Narrative:
  Cars 1, 2, 3 arrive at t=1 and fill all three ports immediately.
  Car 4 also arrives at t=1 but finds no free port — rejected.
  Car 5 arrives at t=5 and is assigned to port 1, which car 1 has just
  vacated (car 1 has a hard deadline of t=4).

  The LP must therefore:
    - Charge car 1 to its target within the tight 4-step window,
      even though electricity is cheaper later in the day.
    - Balance load across the three simultaneous cars (t=1–4) without
      exceeding the 20 kW grid cap (max simultaneous = 3 × 12.8 = 38.4 kW).
    - Serve the late-arriving car 5 optimally over its shorter dwell (t=5–8).

  The rejected car 4 is not modelled in the LP — assignment decisions are
  made externally before the LP is called.  The LP operates only on the
  four cars that have been allocated a port.

Physical constants:
  V     = 400 V  (all ports)
  I_max = 32 A   (all ports)   → max 12.8 kW per port
  P_max = 20 kW  (grid cap)    → congested whenever 3 ports charge at once
  dt    = 1 h

Electricity price schedule (€/kWh):
  t1–t2 : buy 0.20, sell 0.45  (peak morning — highest margin)
  t3–t4 : buy 0.18, sell 0.38
  t5–t6 : buy 0.15, sell 0.30
  t7–t8 : buy 0.12, sell 0.22  (off-peak — lowest margin)

Car summary (real-world):
  Car 1 | arr=1 | dep=4 | port=1 | s_init= 5 | s_target=20 | tight deadline
  Car 2 | arr=1 | dep=8 | port=2 | s_init=10 | s_target=25 | regular
  Car 3 | arr=1 | dep=8 | port=3 | s_init= 8 | s_target=22 | regular
  Car 4 | arr=1 |  ---  |  ---   | REJECTED — all ports occupied on arrival
  Car 5 | arr=5 | dep=8 | port=1 | s_init= 3 | s_target=15 | late arrival
"""
from __future__ import annotations

from pyomo.environ import value

from milp_ev_opt.optimization.model import build_ev_lp_model
from milp_ev_opt.optimization.solver import solve

# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------

_T = 8

# The LP models only the 4 assigned cars (LP indices 1–4).
# Real-world car 4 is rejected before the LP is called.
# Real-world car 5 becomes LP car 4 (arrives at t=5, reuses port 1).
_LP_CAR_LABELS = {
    1: "Car 1  arr=t1  dep=t4  port=1  (tight deadline)",
    2: "Car 2  arr=t1  dep=t8  port=2",
    3: "Car 3  arr=t1  dep=t8  port=3",
    4: "Car 5  arr=t5  dep=t8  port=1  (late arrival, reuses port 1)",
}
_REJECTED = "Car 4  arr=t1  -- REJECTED: all ports occupied on arrival"

_DATA = {
    "J":       3,
    "T":       _T,
    "I":       4,              # 4 LP cars (real-world cars 1, 2, 3, 5)
    "delta_t": 1.0,            # hours per time step

    "P_max":   20_000.0,       # W — tight: 3 × 12.8 kW = 38.4 kW > 20 kW

    "V":     {1: 400.0, 2: 400.0, 3: 400.0, 4: 400.0},  # 4 = BESS port (disabled)
    "I_max": {1: 32.0,  2: 32.0,  3: 32.0},

    # BESS disabled
    "I_high":    0.0,
    "I_low":     0.0,
    "SoCB_init": 0.0,
    "SoCB_min":  0.0,
    "SoCB_max":  0.0,
    "r_bess_ch":  {t: 1.0 for t in range(1, _T + 1)},
    "r_bess_dis": {t: 1.0 for t in range(1, _T + 1)},

    "p_buy":  {1: 0.20, 2: 0.20, 3: 0.18, 4: 0.18,
               5: 0.15, 6: 0.15, 7: 0.12, 8: 0.12},
    "p_sell": {1: 0.45, 2: 0.45, 3: 0.38, 4: 0.38,
               5: 0.30, 6: 0.30, 7: 0.22, 8: 0.22},
    "L":      {t: 0.0 for t in range(1, _T + 1)},

    # Car 1 (LP) occupies port 1 for t=1..4; car 4 (LP) reuses it for t=5..8.
    "assignments": {1: 1, 2: 2, 3: 3, 4: 1},
    "arr": {1: 1, 2: 1, 3: 1, 4: 5},
    "dep": {1: 4, 2: 8, 3: 8, 4: 8},

    "s_init":    {1:  5.0, 2: 10.0, 3:  8.0, 4:  3.0},
    "s_target":  {1: 20.0, 2: 25.0, 3: 22.0, 4: 15.0},
    "s_cap":     {1: 40.0, 2: 60.0, 3: 60.0, 4: 30.0},
    "s_min":     {1:  0.0, 2:  0.0, 3:  0.0, 4:  0.0},
    "P_car_max": {1: 12_800.0, 2: 12_800.0, 3: 12_800.0, 4: 12_800.0},
    "r_car":     {(i, t): 1.0 for i in range(1, 5) for t in range(1, _T + 1)},
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    data = _DATA
    m = build_ev_lp_model(data)
    solve(m, solver="highs")

    T = data["T"]
    J = data["J"]
    W = 76

    # --- header ---
    print("\n" + "=" * W)
    print("  EV Charging LP — Offline Solution (5 cars, 3 ports, 1 rejected)")
    print("=" * W)

    print("\n  Scenario:")
    for lp_i, label in _LP_CAR_LABELS.items():
        print(f"    [SERVED]   {label}")
    print(f"    [REJECTED] {_REJECTED}")

    total_revenue = sum(
        value(m.I_ev[j, t]) * value(m.V[j]) * (value(m.delta_t) / 1000.0)
        * value(m.p_sell[t]) * value(m.z[j, t])
        for j in m.J_ev for t in m.T
    )
    total_cost = sum(
        value(m.I_ev[j, t]) * value(m.V[j]) * (value(m.delta_t) / 1000.0)
        * value(m.p_buy[t])
        for j in m.J_ev for t in m.T
    )
    print(f"\n  Revenue    : {total_revenue:.2f} EUR")
    print(f"  Grid cost  : {total_cost:.2f} EUR")
    print(f"  Net profit : {total_revenue - total_cost:.2f} EUR")

    # --- charging schedule (kW per port) ---
    col = 7
    print(f"\n  Charging schedule (kW per port):")
    header = f"  {'Port':<6}" + "".join(f"  t{t:<{col-2}}" for t in range(1, T + 1))
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)
    for j in range(1, J + 1):
        row = f"  {j:<6}"
        for t in range(1, T + 1):
            kw = value(m.I_ev[j, t]) * value(m.V[j]) / 1000.0
            row += f"  {kw:>{col-2}.2f}"
        print(row)

    # Grid draw per step
    print(sep)
    row = f"  {'total':<6}"
    for t in range(1, T + 1):
        total_kw = sum(
            value(m.I_ev[j, t]) * value(m.V[j]) / 1000.0
            for j in range(1, J + 1)
        )
        row += f"  {total_kw:>{col-2}.2f}"
    print(row)
    print(f"  {'':6}  (grid cap = {data['P_max']/1000:.0f} kW)")

    # --- SoC trajectory (kWh per LP car) ---
    print(f"\n  SoC trajectory (kWh)  [* = target reached]:")
    header2 = f"  {'Car':<6}" + "".join(f"  t{t:<{col-2}}" for t in range(1, T + 1))
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for i in range(1, data["I"] + 1):
        label = _LP_CAR_LABELS[i].split()[0] + " " + _LP_CAR_LABELS[i].split()[1]
        row = f"  {label:<6}"
        for t in range(1, T + 1):
            s = value(m.soc_car[i, t])
            flag = "*" if s >= data["s_target"][i] - 1e-3 else " "
            row += f"  {s:>{col-3}.1f}{flag}"
        print(row)
        dep_label = f"dep=t{data['dep'][i]}" if data["dep"][i] < T else ""
        print(f"  {'':6}  target={data['s_target'][i]:.0f} kWh  "
              f"init={data['s_init'][i]:.0f} kWh  {dep_label}")

    print("\n" + "=" * W + "\n")


if __name__ == "__main__":
    main()
