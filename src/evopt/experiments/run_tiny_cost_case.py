"""
run_tiny_cost_case.py — Small reference scenario for the EV charging LP
========================================================================
Setup:
  3 ports,  8 time steps (1 h each),  3 cars

Narrative:
  All three cars arrive at t=1 with different energy needs and the same
  deadline (t=8).  The grid cap (20 kW) is tight: if all three ports
  charge simultaneously at full rate (3 × 12.8 kW = 38.4 kW) the
  constraint is violated.  The LP must therefore spread load across time.

  Electricity is cheapest in steps 7–8 and most expensive in steps 1–2.
  The LP balances two forces:
    - Charge early to earn the high sell price.
    - Defer to the cheap buy window to reduce cost.
  Because p_sell >> p_buy throughout, revenue dominates and the LP
  charges as fast as the grid cap and car needs allow.

Physical constants:
  V     = 400 V  (all ports)
  I_max = 32 A   (all ports)   → max 12.8 kW per port
  P_max = 20 kW  (grid cap)    → congested when all 3 ports charge at once
  dt    = 1 h

Electricity price schedule (€/kWh):
  t1–t2 : buy 0.20, sell 0.45  (peak morning — highest margin)
  t3–t4 : buy 0.18, sell 0.38
  t5–t6 : buy 0.15, sell 0.30
  t7–t8 : buy 0.12, sell 0.22  (off-peak — lowest margin)

Car summary:
  Car 1 | arr=1 | dep=8 | s_init=5  | s_target=38 | needs 33 kWh
  Car 2 | arr=1 | dep=8 | s_init=10 | s_target=20 | needs 10 kWh
  Car 3 | arr=1 | dep=8 | s_init=8  | s_target=22 | needs 14 kWh
"""
from __future__ import annotations

from pyomo.environ import value

from evopt.optimization.model import build_ev_lp_model
from evopt.optimization.solver import solve

# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------

_T = 8

_DATA = {
    "J":       3,
    "T":       _T,
    "I":       3,
    "delta_t": 1.0,           # hours per time step

    "P_max":   20_000.0,      # W — tight: 3 × 12.8 kW = 38.4 kW > 20 kW

    "V":     {1: 400.0, 2: 400.0, 3: 400.0, 4: 400.0},   # 4 = BESS port (disabled)
    "I_max": {1: 32.0,  2: 32.0,  3: 32.0},

    # BESS disabled for this scenario
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

    "assignments": {1: 1, 2: 2, 3: 3},
    "arr": {1: 1, 2: 1, 3: 1},
    "dep": {1: 8, 2: 8, 3: 8},

    "s_init":    {1: 5.0,  2: 10.0, 3: 8.0},
    "s_target":  {1: 38.0, 2: 20.0, 3: 22.0},
    "s_cap":     {1: 80.0, 2: 60.0, 3: 60.0},
    "s_min":     {1: 0.0,  2: 0.0,  3: 0.0},
    "P_car_max": {1: 12_800.0, 2: 12_800.0, 3: 12_800.0},
    "r_car":     {(i, t): 1.0 for i in range(1, 4) for t in range(1, _T + 1)},
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
    W = 72

    # --- header ---
    print("\n" + "=" * W)
    print("  EV Charging LP — Offline Solution")
    print("=" * W)

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
    print(f"  Revenue    : €{total_revenue:.2f}")
    print(f"  Grid cost  : €{total_cost:.2f}")
    print(f"  Net profit : €{total_revenue - total_cost:.2f}")

    # --- charging schedule (kW per port) ---
    col = 7
    print(f"\n  Charging schedule (kW per port):")
    header = f"  {'Port':<6}" + "".join(f"  t{t:<{col-2}}" for t in range(1, T + 1))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j in range(1, J + 1):
        row = f"  {j:<6}"
        for t in range(1, T + 1):
            kw = value(m.I_ev[j, t]) * value(m.V[j]) / 1000.0
            row += f"  {kw:>{col-2}.2f}"
        print(row)

    # --- SoC trajectory (kWh per car) ---
    print(f"\n  SoC trajectory (kWh per car)  [* = target reached]:")
    header2 = f"  {'Car':<6}" + "".join(f"  t{t:<{col-2}}" for t in range(1, T + 1))
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for i in range(1, data["I"] + 1):
        row = f"  {i:<6}"
        for t in range(1, T + 1):
            s = value(m.soc_car[i, t])
            flag = "*" if s >= data["s_target"][i] - 1e-3 else " "
            row += f"  {s:>{col-3}.1f}{flag}"
        print(row)
        print(f"  {'':6}  target={data['s_target'][i]:.1f} kWh")

    print("=" * W + "\n")


if __name__ == "__main__":
    main()
