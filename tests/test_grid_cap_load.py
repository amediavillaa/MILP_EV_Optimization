"""Tests that background load L is enforced in the grid-cap constraint."""
import pytest
from pyomo.environ import value


def test_background_load_limits_ev_charging(minimal_bess_data):
    """Grid cap must deduct background load L before allocating to EVs/BESS.

    With L = P_max at t=1, zero headroom remains.  The LP must set I_ev = 0
    (and I_bess_ch = 0) at t=1.  Without the fix L is silently ignored,
    so the LP charges the car freely.
    """
    from evopt.optimization.model import build_ev_lp_model
    from evopt.optimization.solver import solve

    data = dict(minimal_bess_data)
    data["L"] = {1: data["P_max"], 2: 0.0}   # full background at t=1 only
    data["I_high"] = 0.0   # disable BESS so it cannot compensate via discharge
    data["I_low"]  = 0.0

    m = build_ev_lp_model(data)
    solve(m, solver="highs")

    # At t=1 the background load consumes all of P_max → no EV or BESS charging
    assert value(m.I_ev[1, 1]) == pytest.approx(0.0, abs=1e-6), (
        f"I_ev[1,1]={value(m.I_ev[1,1]):.4f}A; expected 0 when L=P_max. "
        "Background load may not be enforced in grid-cap constraint."
    )
    assert value(m.I_bess_ch[1]) == pytest.approx(0.0, abs=1e-6), (
        "I_bess_ch[1] should be 0 when background load consumes P_max."
    )


def test_zero_background_load_does_not_restrict_charging(minimal_bess_data):
    """With L=0 (default), grid cap is fully available to EVs and BESS."""
    from evopt.optimization.model import build_ev_lp_model
    from evopt.optimization.solver import solve

    m = build_ev_lp_model(minimal_bess_data)
    solve(m, solver="highs")

    total_ev_power_t1 = sum(
        value(m.I_ev[j, 1]) * value(m.V[j]) for j in m.J_ev
    )
    # Car needs 10 kWh, port max 12.8 kW — LP must charge a non-trivial amount
    assert total_ev_power_t1 > 1000.0
