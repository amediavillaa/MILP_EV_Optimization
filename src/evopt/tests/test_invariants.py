"""
Invariant tests — physical and accounting constraints that must hold
across every scenario and both runners.

Parametrised over 3 fixtures × 2 runners = 6 combinations per invariant.
"""

import pytest
from evopt.experiments.runner import run_rolling_horizon, run_equal_allocation

TOL = 1e-2

RUNNERS = [
    pytest.param(lambda d: run_rolling_horizon(d, horizon=3), id="mpc_h3"),
    pytest.param(lambda d: run_equal_allocation(d),           id="equal_alloc"),
]

SCENARIOS = ["single_car", "port_saturation", "sequential_arrival"]


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_soc_never_exceeds_capacity(request, scenario, runner):
    data = request.getfixturevalue(scenario)
    res  = runner(data)
    for i in range(1, data["I"] + 1):
        for soc in res.soc_trajectory[i]:
            assert soc <= data["s_cap"][i] + TOL, (
                f"car {i} SoC {soc:.4f} exceeds capacity {data['s_cap'][i]}"
            )


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_soc_never_negative(request, scenario, runner):
    data = request.getfixturevalue(scenario)
    res  = runner(data)
    for i in range(1, data["I"] + 1):
        for soc in res.soc_trajectory[i]:
            assert soc >= -TOL, f"car {i} SoC went negative: {soc:.4f}"


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_grid_cap_never_violated(request, scenario, runner):
    data  = request.getfixturevalue(scenario)
    res   = runner(data)
    P_max = data["P_max"]
    for t in range(1, data["T"] + 1):
        total = sum(
            res.current_applied.get((j, t), 0.0) * data["V"][j]
            for j in range(1, data["J"] + 1)
        )
        assert total <= P_max + TOL, (
            f"t={t}: grid draw {total:.2f} W exceeds P_max {P_max:.2f} W"
        )


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_profit_identity(request, scenario, runner):
    data = request.getfixturevalue(scenario)
    res  = runner(data)
    assert res.net_profit == pytest.approx(res.revenue - res.cost, abs=1e-6)


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_soc_frozen_after_departure(request, scenario, runner):
    """Once a car departs its SoC must not change in subsequent steps."""
    data = request.getfixturevalue(scenario)
    res  = runner(data)
    for i in res.departed:
        traj      = res.soc_trajectory[i]
        final_soc = res.soc_final[i]
        # find the step when soc first reached its final value (the actual departure step)
        t_freeze = next(
            t for t in range(1, data["T"] + 1)
            if abs(traj[t] - final_soc) < 1e-6
        )
        for t in range(t_freeze, data["T"] + 1):
            assert traj[t] == pytest.approx(final_soc, abs=1e-6), (
                f"car {i} SoC changed after departure: "
                f"t_freeze={t_freeze}, t={t}, soc={traj[t]:.6f} vs {final_soc:.6f}"
            )


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_rejected_cars_never_charged(request, scenario, runner):
    """Rejected cars have no port and their SoC must equal s_init throughout."""
    data = request.getfixturevalue(scenario)
    res  = runner(data)
    for i in res.rejected:
        assert i not in res.assignments
        for soc in res.soc_trajectory[i]:
            assert soc == pytest.approx(data["s_init"][i], abs=1e-6), (
                f"rejected car {i} SoC changed: {soc:.6f}"
            )
