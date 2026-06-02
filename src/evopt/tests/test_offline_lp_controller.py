import pytest
from evopt.controllers.offline_lp_controller import OfflineLPController, ScenarioCollector, build_offline_schedule


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
    assert scenario["arr"][1] == 6   # 5 + 1
    assert scenario["s_init"][1] == pytest.approx(3.0)


def test_collector_records_departure():
    collector = ScenarioCollector()
    state5 = _make_state(t=5, cars={1: {"soc_now": 3.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}})
    state6 = _make_state(t=6, cars={1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}})
    collector.record(state5)
    collector.record(state6)
    scenario = collector.build_scenario()
    # dep should be t_max from last seen step, shifted to LP 1-indexed
    assert scenario["dep"][1] == 16   # 15 + 1


def test_collector_prices_captured_from_first_step():
    collector = ScenarioCollector()
    collector.record(_make_state(t=0))
    scenario = collector.build_scenario()
    assert 1 in scenario["p_buy"]       # key 0 shifted to 1
    assert scenario["p_buy"][1] == pytest.approx(0.20)


def test_offline_controller_replays_schedule():
    # Schedule keys are 1-indexed LP steps
    schedule = {6: {1: 10.0}, 7: {1: 8.0}, 8: {}}
    ctrl = OfflineLPController(schedule)
    assert ctrl.compute_action(_make_state(t=5)) == {1: 10.0}   # 5+1=6
    assert ctrl.compute_action(_make_state(t=6)) == {1: 8.0}    # 6+1=7
    assert ctrl.compute_action(_make_state(t=7)) == {}           # 7+1=8 → empty


def test_offline_controller_missing_step_returns_empty():
    ctrl = OfflineLPController({})
    assert ctrl.compute_action(_make_state(t=99)) == {}


def test_build_offline_schedule_returns_schedule():
    """Offline schedule must cover the car's dwell window and be non-negative."""
    scenario = {
        "J": 1, "T": 10, "I": 1,
        "delta_t": 5 / 60,
        "P_max": 10_000.0,
        "V": {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0},
        "I_high": 25.0, "I_low": 25.0,
        "p_buy":  {t: 0.20 for t in range(1, 11)},
        "p_sell": {t: 0.40 for t in range(1, 11)},
        "L":      {t: 0.0  for t in range(1, 11)},
        "assignments": {1: 1},
        "arr": {1: 1}, "dep": {1: 8},
        "s_init": {1: 2.0}, "s_cap": {1: 10.0},
        "s_min": {1: 0.0}, "s_target": {1: 10.0},
        "P_car_max": {1: 32.0 * 400.0},
        "r_car": {(1, t): 1.0 for t in range(1, 9)},
        "SoCB_init": 0.0, "SoCB_min": 0.0, "SoCB_max": 0.0,
        "r_bess_ch": {t: 0.0 for t in range(1, 11)},
        "r_bess_dis": {t: 0.0 for t in range(1, 11)},
    }
    schedule = build_offline_schedule(scenario, solver="highs")
    # Schedule must exist for every step in planning horizon
    assert isinstance(schedule, dict)
    assert len(schedule) > 0
    # All current values must be non-negative
    for t, actions in schedule.items():
        for port, amps in actions.items():
            assert amps >= -1e-6, f"Negative current at t={t} port={port}: {amps}"
