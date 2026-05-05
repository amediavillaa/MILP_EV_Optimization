import pytest
from evopt.controllers.lp_controller import LPController


def _make_state(t: int, n_cars: int = 1, p_max_w: float = 10_000.0) -> dict:
    """Minimal Chargax-format state with n_cars, each needing 5 kWh."""
    present_cars = {
        i + 1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 60.0, "t_max": t + 24}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    J = max(n_cars, 1)
    p_buy  = {s: 0.20 for s in range(t, t + 50)}
    p_sell = {s: 0.40 for s in range(t, t + 50)}
    return {
        "t":            t,
        "delta_t":      5 / 60,
        "J":            J,
        "P_max":        p_max_w,
        "V":            {j: 400.0 for j in range(1, J + 1)},
        "I_max":        {j: 32.0  for j in range(1, J + 1)},
        "p_buy":        p_buy,
        "p_sell":       p_sell,
        "present_cars": present_cars,
        "assignments":  assignments,
        "departed_socs": [],
    }


def test_reset_clears_plan():
    ctrl = LPController(horizon_steps=12, solver="highs")
    ctrl._plan = {0: {1: 10.0}}
    ctrl.reset()
    assert ctrl._plan is None


def test_no_cars_returns_empty():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state = _make_state(0, n_cars=0)
    assert ctrl.compute_action(state) == {}


def test_with_one_car_returns_nonempty_actions():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state = _make_state(0, n_cars=1)
    actions = ctrl.compute_action(state)
    assert 1 in actions
    assert actions[1] >= 0.0


def test_re_solves_every_timestep():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    plan_t0 = ctrl._plan

    state1 = _make_state(1)
    ctrl.compute_action(state1)
    assert ctrl._plan is not plan_t0   # new plan object each timestep
