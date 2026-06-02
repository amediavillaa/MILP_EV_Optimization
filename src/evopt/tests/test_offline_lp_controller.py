import pytest
from evopt.controllers.offline_lp_controller import OfflineLPController, ScenarioCollector


def _make_state(t: int, cars: dict | None = None) -> dict:
    """Minimal state dict for testing."""
    if cars is None:
        cars = {1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": t + 10}}
    assignments = {car_id: car_id for car_id in cars}
    J = max(cars) if cars else 1
    return {
        "t": t,
        "delta_t": 5 / 60,
        "J": J,
        "P_max": 10_000.0,
        "V": {j: 400.0 for j in range(1, J + 2)},
        "I_max": {j: 32.0 for j in range(1, J + 1)},
        "p_buy":  {s: 0.20 for s in range(t, t + 300)},
        "p_sell": {s: 0.40 for s in range(t, t + 300)},
        "present_cars": cars,
        "assignments": assignments,
        "departed_fulfillments": [],
        "socb_now": 15.0,
    }


def test_collector_records_arrival():
    collector = ScenarioCollector()
    collector.record(_make_state(t=5, cars={1: {"soc_now": 3.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}}))
    scenario = collector.build_scenario()
    assert scenario["arr"][1] == 5
    assert scenario["s_init"][1] == pytest.approx(3.0)


def test_collector_records_departure():
    collector = ScenarioCollector()
    state5 = _make_state(t=5, cars={1: {"soc_now": 3.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}})
    state6 = _make_state(t=6, cars={1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}})
    collector.record(state5)
    collector.record(state6)
    scenario = collector.build_scenario()
    # dep should be t_max from last seen step
    assert scenario["dep"][1] == 15


def test_collector_prices_captured_from_first_step():
    collector = ScenarioCollector()
    collector.record(_make_state(t=0))
    scenario = collector.build_scenario()
    assert 0 in scenario["p_buy"]
    assert scenario["p_buy"][0] == pytest.approx(0.20)


def test_offline_controller_replays_schedule():
    schedule = {5: {1: 10.0}, 6: {1: 8.0}, 7: {}}
    ctrl = OfflineLPController(schedule)
    assert ctrl.compute_action(_make_state(t=5)) == {1: 10.0}
    assert ctrl.compute_action(_make_state(t=6)) == {1: 8.0}
    assert ctrl.compute_action(_make_state(t=7)) == {}


def test_offline_controller_missing_step_returns_empty():
    ctrl = OfflineLPController({})
    assert ctrl.compute_action(_make_state(t=99)) == {}
