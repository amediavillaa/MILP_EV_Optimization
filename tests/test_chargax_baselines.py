import pytest
from milp_ev_opt.controllers.chargax_baselines import MaxChargeController, RandomController


def _make_state(n_cars: int = 2) -> dict:
    present_cars = {
        i + 1: {"soc_now": 10.0, "s_target": 30.0, "s_cap": 60.0, "t_max": 100}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    return {
        "t": 0, "delta_t": 5/60, "J": n_cars,
        "P_max": 20_000.0,
        "V":    {j: 400.0 for j in range(1, n_cars + 1)},
        "I_max": {j: 32.0  for j in range(1, n_cars + 1)},
        "p_buy": {0: 0.20}, "p_sell": {0: 0.40},
        "present_cars": present_cars,
        "assignments":  assignments,
        "departed_fulfillments": [],
    }


def test_max_charge_returns_i_max_for_each_port():
    actions = MaxChargeController().compute_action(_make_state(2))
    assert actions[1] == pytest.approx(32.0)
    assert actions[2] == pytest.approx(32.0)


def test_max_charge_empty_state_returns_empty():
    assert MaxChargeController().compute_action(_make_state(0)) == {}


def test_random_returns_non_negative_actions():
    actions = RandomController(seed=0).compute_action(_make_state(2))
    assert all(v >= 0 for v in actions.values())


def test_random_respects_i_max():
    actions = RandomController(seed=42).compute_action(_make_state(2))
    assert all(v <= 32.0 for v in actions.values())


def test_both_controllers_reset_without_error():
    MaxChargeController().reset()
    RandomController().reset()
