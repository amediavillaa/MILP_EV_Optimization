"""Tests for compute_summary_metrics."""
import pytest
from evopt.benchmarking.results import ChargaxSimResults
from evopt.metrics.evaluation import compute_summary_metrics


def _r(name: str, profit: float, served: int, rejected: int,
       fulfillment: float = 0.9) -> ChargaxSimResults:
    return ChargaxSimResults(
        controller_name=name, seed=0, episode_date="2023-01-01",
        net_profit=profit, total_revenue=profit + 2, total_cost=2.0,
        served_customers=served, rejected_customers=rejected,
        mean_soc_fulfillment=fulfillment,
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
    results = [_r("milp", 10.0, 8, 2)]
    summary = compute_summary_metrics(results)
    assert summary["milp"]["served_rate_mean"] == pytest.approx(0.8)


def test_returns_mean_soc_fulfillment_not_departure_soc():
    """compute_summary_metrics must aggregate mean_soc_fulfillment (renamed field)."""
    results = [_r("lp", 5.0, 3, 0, fulfillment=0.75),
               _r("lp", 6.0, 4, 0, fulfillment=0.85)]
    summary = compute_summary_metrics(results)
    assert "mean_soc_fulfillment" in summary["lp"], (
        "Key 'mean_soc_fulfillment' missing — field may still reference "
        "old name 'mean_soc_at_departure'."
    )
    assert summary["lp"]["mean_soc_fulfillment"] == pytest.approx(0.80)
