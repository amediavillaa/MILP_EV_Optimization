"""
run_tiny_cost_case.py — Small reference scenario for the EV charging LP
========================================================================
Setup:
  3 ports,  8 time steps (1 h each),  5 cars

Narrative:
  Cars 1, 2, 3 arrive at t=1 and fill all three ports immediately.
  Car 4 also arrives at t=1 but finds no free port → rejected.
  Car 5 arrives at t=4.

  Revenue is not known at arrival. Instead it is earned per kWh
  delivered, at the sell price prevailing in each time step.  The
  optimiser therefore balances:
    - Charging cars earlier (higher sell price at t1–t2) vs.
    - Spreading load to avoid hitting the grid cap (P_max = 20 kW).

  Car 1 needs 33 kWh to reach s_target; aggressive charging in
  t=1..3 frees its port for car 5 at t=4.  Whether this trade-off
  is worth it depends entirely on the sell/buy price spread across
  the horizon — not on any fixed per-car revenue figure.

Physical constants:
  V     = 400 V  (all ports)
  I_max = 32 A   (all ports)   → max 12.8 kW per port
  P_max = 20 kW  (grid cap)    → congested when 3 cars charge simultaneously
  dt    = 1 h

Electricity price schedule (€/kWh):
  t1–t2 : buy 0.20, sell 0.45  (peak morning — highest margin)
  t3–t4 : buy 0.18, sell 0.38
  t5–t6 : buy 0.15, sell 0.30
  t7–t8 : buy 0.12, sell 0.22  (off-peak — lowest margin)

Car summary:
  Car 1 | arr=1 | s_init=5  | s_target=38 | needs aggressive charging
  Car 2 | arr=1 | s_init=10 | s_target=20 | regular, can wait
  Car 3 | arr=1 | s_init=8  | s_target=22 | regular, can wait
  Car 4 | arr=1 | s_init=20 | s_target=35 | REJECTED (no port)
  Car 5 | arr=4 | s_init=12 | s_target=30 | only served if car 1 clears first
"""

SAMPLE_DATA = {
    # ------------------------------------------------------------------ topology
    "J" : 3,               # ports 
    "T" : 8,               # time 
    "I" : 5,               # cars 

    # ------------------------------------------------------------------ time
    "delta_t" : 1.0,       # hours per time step

    # ------------------------------------------------------------------ grid
    "P_max" : 20_000.0,    # W — deliberately tight to create congestion

    # ------------------------------------------------------------------ solver
    # M_big >= max(s_cap) is a safe upper bound.
    # For tighter LP relaxation use per-car M_i = s_cap[i] in production.
    "M_big"   : 100.0,
    "epsilon" : 0.1,        # kWh — tolerance for departure trigger

    # ------------------------------------------------------------------ ports (1-indexed)
    "V"    : {1: 400.0, 2: 400.0, 3: 400.0},   # V
    "I_max": {1: 32.0,  2: 32.0,  3: 32.0},    # A  → 12.8 kW max per port

    # ------------------------------------------------------------------ electricity prices (€/kWh)
    "p_buy" : {1: 0.20, 2: 0.20, 3: 0.18, 4: 0.18,
               5: 0.15, 6: 0.15, 7: 0.12, 8: 0.12},
    "p_sell": {1: 0.45, 2: 0.45, 3: 0.38, 4: 0.38,
               5: 0.30, 6: 0.30, 7: 0.22, 8: 0.22},

    # ------------------------------------------------------------------ cars (1-indexed)
    "arr"  : {1: 1, 2: 1, 3: 1, 4: 1, 5: 4},
    "t_max": {1: 8, 2: 8, 3: 8, 4: 8, 5: 8},

    # Initial SoC on arrival (kWh)
    "s_init": {1: 5.0, 2: 10.0, 3: 8.0, 4: 20.0, 5: 12.0},

    # Target SoC — car departs once reached (kWh)
    # Car 1 needs 33 kWh; reachable in 3 steps only with priority current (~27.5 A/step)
    "s_target": {1: 38.0, 2: 20.0, 3: 22.0, 4: 35.0, 5: 30.0},

    "s_cap": {1: 80.0, 2: 60.0, 3: 60.0, 4: 60.0, 5: 60.0},
}
