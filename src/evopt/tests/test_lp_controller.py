import pytest
from evopt.controllers.lp_controller import LPController


def _make_state(t: int, n_cars: int = 1, p_max_w: float = 10_000.0,
                socb_now: float = 15.0) -> dict:
    """Minimal Chargax-format state with n_cars and BESS."""
    present_cars = {
        i + 1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 60.0, "t_max": t + 24}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    J = max(n_cars, 1)
    p_buy  = {s: 0.20 for s in range(t, t + 100)}
    p_sell = {s: 0.40 for s in range(t, t + 100)}
    return {
        "t":             t,
        "delta_t":       5 / 60,
        "J":             J,
        "P_max":         p_max_w,
        "V":             {j: 400.0 for j in range(1, J + 1)},
        "I_max":         {j: 32.0  for j in range(1, J + 1)},
        "p_buy":         p_buy,
        "p_sell":        p_sell,
        "present_cars":  present_cars,
        "assignments":   assignments,
        "departed_socs": [],
        "socb_now":      socb_now,
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


def test_bess_action_in_plan():
    """BESS port J+1 must appear in the plan."""
    ctrl = LPController(horizon_steps=6, solver="highs")
    state = _make_state(0, n_cars=1)
    ctrl.compute_action(state)
    J = state["J"]
    t = state["t"]
    assert (J + 1) in ctrl._plan[t]


def test_plan_cached_within_horizon():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    plan_after_t0 = ctrl._plan

    state1 = _make_state(1)
    ctrl.compute_action(state1)
    assert ctrl._plan is plan_after_t0   # no re-solve between horizon boundaries


def test_resolves_at_horizon_boundary():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    plan_after_t0 = ctrl._plan

    state12 = _make_state(12)
    ctrl.compute_action(state12)
    assert ctrl._plan is not plan_after_t0   # new solve at t=12


def test_resolves_when_plan_is_none():
    ctrl = LPController(horizon_steps=12, solver="highs")
    ctrl._plan = None
    state = _make_state(5)
    ctrl.compute_action(state)
    assert ctrl._plan is not None


def test_build_lp_data_has_required_keys():
    ctrl = LPController(horizon_steps=6, solver="highs")
    state = _make_state(0, n_cars=1)
    data = ctrl._build_lp_data(state)
    required = {
        "J", "T", "delta_t", "P_max", "V", "I_max",
        "I_high", "I_low", "p_buy", "p_sell", "L",
        "assignments", "dep", "s_cap", "s_min", "P_car_max",
        "r_car", "SoCB_init", "SoCB_min", "SoCB_max",
        "r_bess_ch", "r_bess_dis",
    }
    assert required <= data.keys()


def test_build_lp_data_bess_port_in_V():
    ctrl = LPController(horizon_steps=6, solver="highs")
    state = _make_state(0, n_cars=1)
    data = ctrl._build_lp_data(state)
    J = state["J"]
    assert (J + 1) in data["V"]
    assert data["V"][J + 1] == pytest.approx(ctrl.v_bess)
