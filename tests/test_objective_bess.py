import pytest
from pyomo.environ import value


def test_model_has_no_p_sell_bess(minimal_bess_data):
    """After fix, the LP model should not carry a p_sell_bess parameter."""
    from evopt.optimization.model import build_ev_lp_model

    m = build_ev_lp_model(minimal_bess_data)
    assert not hasattr(m, "p_sell_bess")


def test_bess_discharge_reduces_objective_value(minimal_bess_data):
    """
    Discharging BESS at the expensive step (t=2, p_buy=0.30) should reduce
    the negated-profit objective compared to no discharge.

    Under the corrected grid_cost formula, each kWh discharged at t=2 saves
    0.30 in grid cost → more profit → lower objective (sense=minimize).
    """
    from evopt.optimization.model import build_ev_lp_model

    m = build_ev_lp_model(minimal_bess_data)

    for j in m.J_ev:
        for t in m.T:
            m.I_ev[j, t].fix(10.0)
    for t in m.T:
        m.I_bess_ch[t].fix(0.0)
        m.I_bess_dis[t].fix(0.0)
        m.SoCB[t].fix(5.0)
    for i in m.I:
        for t in m.T:
            m.soc_car[i, t].fix(2.0)

    obj_no_discharge = value(m.obj)

    m.I_bess_dis[2].fix(5.0)    # discharge 5 A at expensive t=2
    obj_with_discharge = value(m.obj)

    # More BESS discharge → less grid cost → more profit → lower negated obj
    assert obj_with_discharge < obj_no_discharge
