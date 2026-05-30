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


def test_table_scenario_csv(tmp_path):
    from evopt.experiments.tiny_case_figures import write_table_scenario
    write_table_scenario(tmp_path)
    text = (tmp_path / "table_scenario.csv").read_text()
    assert "Car 4" in text
    assert "Rejected" in text
    assert "20" in text   # s_target for Car 1
    assert "Port" in text


def test_table_scenario_tex(tmp_path):
    from evopt.experiments.tiny_case_figures import write_table_scenario
    write_table_scenario(tmp_path)
    tex = (tmp_path / "table_scenario.tex").read_text()
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert "Rejected" in tex
    assert "Car 4" in tex


def test_table_solution_csv_metrics(tmp_path, res):
    from evopt.experiments.tiny_case_figures import write_table_solution
    write_table_solution(res, tmp_path)
    text = (tmp_path / "table_solution.csv").read_text()
    assert "Revenue" in text
    assert "54" in text
    assert "26" in text
    assert "28" in text


def test_table_solution_tex(tmp_path, res):
    from evopt.experiments.tiny_case_figures import write_table_solution
    write_table_solution(res, tmp_path)
    tex = (tmp_path / "table_solution.tex").read_text()
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert "Revenue" in tex


def test_soc_trajectories_pdf(tmp_path, res):
    from evopt.experiments.tiny_case_figures import plot_soc_trajectories
    plot_soc_trajectories(res, tmp_path)
    pdf = tmp_path / "soc_trajectories.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_charging_schedule_pdf(tmp_path, res):
    from evopt.experiments.tiny_case_figures import plot_charging_schedule
    plot_charging_schedule(res, tmp_path)
    pdf = tmp_path / "charging_schedule.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
    assert pdf.read_bytes()[:4] == b"%PDF"
