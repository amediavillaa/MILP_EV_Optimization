import pytest
from milp_ev_opt.controllers.equal_share import EqualShareController


def _make_state(n_cars: int, p_max_w: float = 10_000.0) -> dict:
    """Build a minimal Chargax-format state dict with n_cars on sequential ports."""
    present_cars = {
        i + 1: {"soc_now": 10.0, "s_target": 30.0, "s_cap": 60.0, "t_max": 100}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    return {
        "t": 0,
        "delta_t": 5 / 60,
        "J": max(n_cars, 1),
        "P_max": p_max_w,
        "V":    {j: 400.0 for j in range(1, max(n_cars, 1) + 1)},
        "I_max": {j: 32.0  for j in range(1, max(n_cars, 1) + 1)},
        "p_buy":  {0: 0.20},
        "p_sell": {0: 0.40},
        "present_cars": present_cars,
        "assignments": assignments,
        "departed_socs": [],
    }


def test_no_cars_returns_empty():
    ctrl = EqualShareController()
    state = _make_state(0)
    assert ctrl.compute_action(state) == {}


def test_single_car_gets_all_capacity():
    ctrl = EqualShareController()
    state = _make_state(1, p_max_w=4_000.0)
    actions = ctrl.compute_action(state)
    # P_max=4000W / V=400V = 10A, I_max=32A → not capped
    assert actions[1] == pytest.approx(10.0, abs=0.01)


def test_two_cars_split_equally():
    ctrl = EqualShareController()
    state = _make_state(2, p_max_w=8_000.0)
    actions = ctrl.compute_action(state)
    # P_share = 4000W; I = 4000/400 = 10A per car
    assert actions[1] == pytest.approx(10.0, abs=0.01)
    assert actions[2] == pytest.approx(10.0, abs=0.01)


def test_reset_does_not_raise():
    EqualShareController().reset()
