"""Tests for BESS eta (efficiency) model correctness.

The BESS efficiency model must match Chargax semantics:
  - SoC dynamics: raw commanded current (no eta factor in the SoC update)
  - Grid cost: eta-adjusted — charging draws I_ch/eta from grid,
    discharging saves I_dis*eta from grid

This ensures the LP's internal model is consistent and its SoC predictions
match the Chargax simulation across the receding horizon.
"""
import pytest
from pyomo.environ import value


def test_bess_soc_uses_raw_current_when_charging(minimal_bess_data):
    """BESS SoC constraint must use raw commanded current, not eta-scaled current.

    With I_high=10A, V=400V, delta_t=1h:
      raw net_in  = 10 × 400 / 1000 = 4.0 kWh  → SoCB[1] = 14.0  (correct)
      eta net_in  = 10 × 0.95 × 400 / 1000 = 3.8 kWh → SoCB[1] ≤ 13.8 (wrong)

    Set SoCB_min = 13.9 (above eta max, below raw max). LP must be feasible
    only if using raw current; infeasible if eta is applied to SoC.
    """
    from evopt.optimization.model import build_rolling_model
    from evopt.optimization.solver import solve
    from pyomo.opt import TerminationCondition

    data = dict(minimal_bess_data)
    data["I_high"]   = 10.0
    data["SoCB_min"] = 13.9   # > 10 + 10×0.95×0.4 = 13.8, < 10 + 10×0.4 = 14.0
    data["SoCB_max"] = 20.0

    m = build_rolling_model(
        data, t_start=1, horizon=1,
        assignments={1: 1},
        soc_now={1: 0.0},
        socb_now=10.0,
        bare=True,
    )
    # With raw current: 10A x 400V x 1h / 1000 = 4 kWh -> 10+4=14 >= 13.9 -> OPTIMAL
    # With eta=0.95:   10A x 0.95 x 400V x 1h / 1000 = 3.8 kWh -> only 13.8 < 13.9 -> INFEASIBLE
    try:
        result = solve(m, solver="highs")
    except Exception as exc:
        pytest.fail(
            f"LP raised {type(exc).__name__} (infeasible). "
            "SoC constraint may apply eta to current -- max SoCB is 13.8 < SoCB_min=13.9."
        )
    tc = result.solver.termination_condition
    assert tc == TerminationCondition.optimal, (
        f"LP is {tc}; expected optimal. "
        "SoC constraint may apply eta to current (SoCB would only reach 13.8 < 13.9)."
    )


def test_bess_charging_grid_cost_includes_eta_penalty(minimal_bess_data):
    """Charging BESS by I_ch amps must draw I_ch/eta amps-equivalent from the grid.

    With I_bess_ch[1]=10A, V=400V, p_buy=0.10, delta_t=1h, eta=0.95:
      expected grid cost = 10/0.95 × 400 × 0.10 × 1.0 / 1000 ≈ 0.42105
      without eta:         10        × 400 × 0.10 × 1.0 / 1000  = 0.40000
    """
    from evopt.optimization.model import build_ev_lp_model

    m = build_ev_lp_model(minimal_bess_data)

    for j in m.J_ev:
        for t in m.T:
            m.I_ev[j, t].fix(0.0)
    for t in m.T:
        m.I_bess_ch[t].fix(0.0)
        m.I_bess_dis[t].fix(0.0)
        m.SoCB[t].fix(10.0)
    for i in m.I:
        for t in m.T:
            m.soc_car[i, t].fix(0.0)

    m.I_bess_ch[1].fix(10.0)   # charge 10A at t=1 (p_buy=0.10)

    obj_val = value(m.obj)

    eta = 0.95
    expected = 10.0 / eta * 400.0 * 0.10 * 1.0 / 1000.0
    assert obj_val == pytest.approx(expected, rel=1e-4), (
        f"Expected charging grid cost ≈{expected:.5f} (I/eta factor), got {obj_val:.5f}. "
        "Eta may not be applied to BESS charging in the objective."
    )


def test_bess_discharging_grid_saving_includes_eta_reduction(minimal_bess_data):
    """Discharging BESS by I_dis amps saves only eta×I_dis amps-equivalent from the grid.

    With I_bess_dis[2]=10A, V=400V, p_buy=0.30, delta_t=1h, eta=0.95:
      expected grid saving = eta × 10 × 400 × 0.30 × 1.0 / 1000 = 1.14
      without eta:                 10 × 400 × 0.30 × 1.0 / 1000  = 1.20
    Objective = -(revenue - grid_cost) = grid_cost = -saving (negative = profit gained).
    """
    from evopt.optimization.model import build_ev_lp_model

    m = build_ev_lp_model(minimal_bess_data)

    for j in m.J_ev:
        for t in m.T:
            m.I_ev[j, t].fix(0.0)
    for t in m.T:
        m.I_bess_ch[t].fix(0.0)
        m.I_bess_dis[t].fix(0.0)
        m.SoCB[t].fix(10.0)
    for i in m.I:
        for t in m.T:
            m.soc_car[i, t].fix(0.0)

    m.I_bess_dis[2].fix(10.0)   # discharge 10A at t=2 (p_buy=0.30)

    obj_val = value(m.obj)

    eta = 0.95
    # grid_cost = -eta * I_dis * V * p_buy * dt / 1000  (saving = negative cost)
    # obj = -(revenue - grid_cost) = -(-eta*I*V*p/1000) = -eta*I*V*p/1000 (negative)
    expected = -(eta * 10.0 * 400.0 * 0.30 * 1.0 / 1000.0)
    assert obj_val == pytest.approx(expected, rel=1e-4), (
        f"Expected discharge grid saving ≈{expected:.5f} (eta×I factor), got {obj_val:.5f}. "
        "Eta may not be applied to BESS discharging in the objective."
    )
