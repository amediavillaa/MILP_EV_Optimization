from __future__ import annotations
import pytest
from pathlib import Path

from evopt.experiments.run_tiny_cost_case import _DATA
from evopt.optimization.model import build_ev_lp_model
from evopt.optimization.solver import solve


@pytest.fixture(scope="module")
def res():
    from evopt.experiments.tiny_case_figures import extract_results
    m = build_ev_lp_model(_DATA)
    solve(m, solver="highs")
    return extract_results(m, _DATA)


def test_revenue(res):
    assert abs(res.revenue - 54.0) < 0.01

def test_cost(res):
    assert abs(res.cost - 26.0) < 0.01

def test_profit(res):
    assert abs(res.profit - 28.0) < 0.01

def test_grid_cap_never_exceeded(res):
    for t in range(1, 9):
        total = sum(res.power[j, t] for j in [1, 2, 3])
        assert total <= 20.0 + 1e-3, f"Grid cap exceeded at t={t}: {total:.3f} kW"

def test_car1_target_reached_by_departure(res):
    # Car 1 (LP i=1) must hit s_target=20 kWh by t=4
    assert res.soc[1, 4] >= 20.0 - 1e-3

def test_car5_target_reached(res):
    # Car 5 (LP i=4) must hit s_target=15 kWh by t=8
    assert res.soc[4, 8] >= 15.0 - 1e-3
