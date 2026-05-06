import pytest
from evopt.benchmarking.results import ChargaxSimResults, StepRecord
from evopt.metrics.evaluation import compute_summary_metrics


def _r(name, profit, served, rejected):
    return ChargaxSimResults(
        controller_name=name, seed=0, episode_date="2023-01-01",
        net_profit=profit, total_revenue=profit+2, total_cost=2,
        served_customers=served, rejected_customers=rejected,
        mean_soc_at_departure=0.9,
        step_log=[],
    )


def test_groups_by_controller():
    results = [
        _r("milp", 10.0, 8, 1), _r("milp", 12.0, 9, 0),
        _r("equal", 7.0, 6, 2), _r("equal", 8.0, 7, 1),
    ]
    summary = compute_summary_metrics(results)
    assert "milp" in summary
    assert "equal" in summary


def test_computes_mean_profit():
    results = [_r("milp", 10.0, 8, 1), _r("milp", 12.0, 9, 0)]
    summary = compute_summary_metrics(results)
    assert summary["milp"]["net_profit_mean"] == pytest.approx(11.0)


def test_computes_std_profit():
    results = [_r("milp", 10.0, 8, 1), _r("milp", 12.0, 9, 0)]
    summary = compute_summary_metrics(results)
    assert summary["milp"]["net_profit_std"] == pytest.approx(1.0)


def test_computes_served_rate():
    results = [_r("milp", 10.0, 8, 2)]   # 8 served, 2 rejected → rate = 0.8
    summary = compute_summary_metrics(results)
    assert summary["milp"]["served_rate_mean"] == pytest.approx(0.8)
