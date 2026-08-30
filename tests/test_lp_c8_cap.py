"""Regression tests for rolling LP C8 fix: SoC upper bound must use s_cap, not s_target.

Before the fix, C8 in add_rolling_constraints capped soc_car at s_target.  The LP
therefore stopped charging each car once the customer's requested level was reached,
leaving the port idle for the rest of the horizon.  Simpler baselines (max_charge)
charge to physical battery capacity (s_cap), so they deliver more energy per dwell
and achieve better port turnover — making the LP look worse than a greedy heuristic.

The fix changes the C8 upper bound from data["s_target"][i] to m.s_cap[i].  The
must_serve constraints still correctly reference s_target as the minimum target to
achieve; C8 now only prevents overcharging beyond the physical battery maximum.
"""
from __future__ import annotations

import pytest


def test_lp_charges_past_s_target_when_profitable():
    """LP must plan charging beyond s_target when p_sell > p_buy and s_cap allows."""
    from milp_ev_opt.controllers.lp_controller import LPController

    ctrl = LPController(horizon_steps=12, socb_min=0.0, socb_max=30.0)
    state = {
        "t": 1,
        "J": 1,
        "delta_t": 5 / 60.0,
        "P_max": 20_000.0,
        "V": {1: 400.0},
        "I_max": {1: 32.0},
        "I_high": 0.0,   # BESS disabled to isolate EV charging behaviour
        "I_low":  0.0,
        "p_buy":  {t: 0.20 for t in range(1, 300)},
        "p_sell": {t: 0.75 for t in range(1, 300)},
        "present_cars": {
            0: {"soc_now": 0.0, "s_target": 2.0, "s_cap": 20.0, "t_max": 50},
        },
        "assignments": {0: 1},
    }

    plan = ctrl._solve(state)

    assert plan, "LP returned an empty plan"

    total_kwh = sum(
        actions.get(1, 0.0) * 400.0 * (5 / 60.0) / 1000.0
        for actions in plan.values()
    )
    # s_target=2 kWh, s_cap=20 kWh.  p_sell (0.75) > p_buy (0.20) makes all
    # charging profitable.  Before the fix C8 capped soc_car at s_target, so
    # the LP delivered exactly 2 kWh then went idle.  After the fix the LP
    # charges to the port/grid limit (~12.8 kWh over 12 × 5-min steps at 32 A).
    assert total_kwh > 2.0 + 0.1, (
        f"LP planned {total_kwh:.3f} kWh but expected > s_target=2 kWh. "
        "C8 may still be capping at s_target instead of s_cap."
    )


def test_rolling_model_feasible_with_soc_between_target_and_cap():
    """build_rolling_model must be feasible when soc_now is strictly between s_target and s_cap.

    Old C8 (soc_car <= s_target): C7's initial condition forces
      soc_car[i, t_start] = soc_now + energy_in >= soc_now > s_target → infeasible.
    New C8 (soc_car <= s_cap): soc_now <= s_cap is satisfied → feasible.
    """
    from pyomo.opt import TerminationCondition
    from milp_ev_opt.optimization.model import build_rolling_model
    from milp_ev_opt.optimization.solver import solve

    window = range(1, 13)
    data = {
        "J":          1,
        "T":          20,
        "delta_t":    5 / 60.0,
        "P_max":      20_000.0,
        "V":          {1: 400.0, 2: 400.0},
        "I_max":      {1: 32.0},
        "I_high":     0.0,
        "I_low":      0.0,
        "p_buy":      {t: 0.20 for t in range(1, 300)},
        "p_sell":     {t: 0.75 for t in range(1, 300)},
        "L":          {t: 0.0 for t in window},
        "assignments": {0: 1},
        "dep":        {0: 15},
        "s_cap":      {0: 20.0},
        "s_target":   {0: 5.0},
        "s_min":      {0: 0.0},
        "P_car_max":  {0: 32.0 * 400.0},
        "r_car":      {(0, t): 1.0 for t in window},
        "SoCB_min":   0.0,
        "SoCB_max":   30.0,
        "r_bess_ch":  {t: 1.0 for t in window},
        "r_bess_dis": {t: 1.0 for t in window},
    }

    # soc_now=7 is strictly between s_target=5 and s_cap=20 (not clamped — we
    # bypass LPController to test the model constraint directly).
    m = build_rolling_model(
        data, t_start=1, horizon=12,
        assignments={0: 1},
        soc_now={0: 7.0},
        socb_now=0.0,
        bare=True,
    )
    assert m is not None

    result = solve(m, solver="highs")
    tc = result.solver.termination_condition
    assert tc == TerminationCondition.optimal, (
        f"Expected optimal solution but got {tc}. "
        "C8 may still cap soc_car at s_target=5, making soc_now=7 infeasible."
    )
