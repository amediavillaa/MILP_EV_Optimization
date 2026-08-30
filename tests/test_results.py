import pytest
from milp_ev_opt.benchmarking.results import ChargaxSimResults, StepRecord


def _make_step(t: int) -> StepRecord:
    return StepRecord(
        t=t,
        actions={1: 16.0},
        reward=0.5,
        profit_delta=0.1,
        served=0,
        rejected=0,
        step_revenue=0.08,
        step_cost=0.04,
    )


def test_step_record_fields():
    s = _make_step(0)
    assert s.t == 0
    assert s.actions == {1: 16.0}
    assert s.step_revenue == 0.08


def test_chargax_sim_results_construction():
    result = ChargaxSimResults(
        controller_name="test",
        seed=0,
        episode_date="2023-01-01",
        net_profit=5.0,
        total_revenue=8.0,
        total_cost=3.0,
        served_customers=4,
        rejected_customers=1,
        mean_soc_fulfillment=0.85,
        step_log=[_make_step(0), _make_step(1)],
    )
    assert result.net_profit == 5.0
    assert len(result.step_log) == 2


class MockFinalState:
    timestep = 287
    profit = 12.5
    served_customers = 6
    rejected_customers = 2
    datetime = "2023-03-15"


def test_from_final_state():
    step_log = [_make_step(i) for i in range(3)]
    result = ChargaxSimResults.from_final_state(
        controller_name="milp_h12",
        seed=3,
        state=MockFinalState(),
        step_log=step_log,
        departures_fulfillment=[0.75, 0.50, 1.0],
    )
    assert result.controller_name == "milp_h12"
    assert result.seed == 3
    assert result.net_profit == pytest.approx(3 * 0.08 - 3 * 0.04)
    assert result.served_customers == 6
    assert result.total_revenue == pytest.approx(3 * 0.08)
    assert result.total_cost == pytest.approx(3 * 0.04)
    assert result.mean_soc_fulfillment == pytest.approx((0.75 + 0.50 + 1.0) / 3)
