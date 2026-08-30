"""Tests that LPController clamps soc_now to s_target before each LP solve.

When Chargax discretisation delivers slightly more energy than planned, a car's
soc_now can exceed its s_target.  The rolling model's C8 constraint
(soc_car <= s_target) would then be immediately infeasible, causing _solve to
return {} and silencing charging for every car that step.  The fix clamps
soc_now to s_target so the LP treats an over-charged car as being exactly at
its target and assigns it zero current, leaving other cars unaffected.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


def _make_state(soc_car0: float, soc_car1: float) -> dict:
    """Two-port state with car 0 at soc_car0 (may exceed its target) and car 1 at soc_car1."""
    return {
        "t":     1,
        "J":     2,
        "delta_t": 5 / 60.0,
        "P_max": 18_000.0,
        "V":     {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0,  2: 32.0},
        "I_high": 25.0,
        "I_low":  0.0,
        "p_buy":  {t: 0.20 for t in range(1, 300)},
        "p_sell": {t: 0.75 for t in range(1, 300)},
        "present_cars": {
            0: {"soc_now": soc_car0, "s_target": 10.0, "s_cap": 30.0, "t_max": 20},
            1: {"soc_now": soc_car1, "s_target": 20.0, "s_cap": 30.0, "t_max": 50},
        },
        "assignments": {0: 1, 1: 2},
    }


def test_soc_above_target_is_clamped_before_lp():
    """_solve passes soc_now clamped to s_target when soc_now > s_target."""
    from milp_ev_opt.controllers.lp_controller import LPController

    ctrl = LPController(horizon_steps=2)
    state = _make_state(soc_car0=12.0, soc_car1=5.0)  # car 0 is over-target (12 > 10)

    captured: dict = {}

    def mock_build(data, t_start, horizon, assignments, soc_now, socb_now, bare=False):
        captured["soc_now"] = dict(soc_now)
        return None  # short-circuit LP solve

    with patch("milp_ev_opt.controllers.lp_controller.build_rolling_model", mock_build):
        ctrl._solve(state)

    assert captured["soc_now"][0] == pytest.approx(10.0)   # clamped from 12.0
    assert captured["soc_now"][1] == pytest.approx(5.0)    # unchanged


def test_soc_at_target_is_unchanged():
    """_solve passes soc_now unchanged when soc_now == s_target."""
    from milp_ev_opt.controllers.lp_controller import LPController

    ctrl = LPController(horizon_steps=2)
    state = _make_state(soc_car0=10.0, soc_car1=5.0)  # car 0 exactly at target

    captured: dict = {}

    def mock_build(data, t_start, horizon, assignments, soc_now, socb_now, bare=False):
        captured["soc_now"] = dict(soc_now)
        return None

    with patch("milp_ev_opt.controllers.lp_controller.build_rolling_model", mock_build):
        ctrl._solve(state)

    assert captured["soc_now"][0] == pytest.approx(10.0)   # no change
    assert captured["soc_now"][1] == pytest.approx(5.0)


def test_soc_below_target_is_unchanged():
    """_solve passes soc_now unchanged when soc_now < s_target (normal case)."""
    from milp_ev_opt.controllers.lp_controller import LPController

    ctrl = LPController(horizon_steps=2)
    state = _make_state(soc_car0=3.0, soc_car1=5.0)

    captured: dict = {}

    def mock_build(data, t_start, horizon, assignments, soc_now, socb_now, bare=False):
        captured["soc_now"] = dict(soc_now)
        return None

    with patch("milp_ev_opt.controllers.lp_controller.build_rolling_model", mock_build):
        ctrl._solve(state)

    assert captured["soc_now"][0] == pytest.approx(3.0)
    assert captured["soc_now"][1] == pytest.approx(5.0)
