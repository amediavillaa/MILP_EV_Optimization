"""
Edge case tests — robustness under unusual inputs.
Each test defines its own minimal scenario locally.
"""

import pytest
from evopt.experiments.runner import run_rolling_horizon, run_equal_allocation


def _base(overrides: dict) -> dict:
    """Minimal 1-port, 1-car, 3-step scenario with selected keys overridden."""
    data = {
        "J": 1, "T": 3, "I": 1,
        "delta_t": 1.0,
        "P_max": 5_000.0,
        "M_big": 50.0,
        "epsilon": 0.05,
        "V":     {1: 400.0},
        "I_max": {1: 10.0},
        "p_buy":  {1: 0.10, 2: 0.10, 3: 0.10},
        "p_sell": {1: 0.20, 2: 0.20, 3: 0.20},
        "arr":    {1: 1},
        "t_max":  {1: 3},
        "s_init":   {1: 5.0},
        "s_target": {1: 10.0},
        "s_cap":    {1: 20.0},
    }
    data.update(overrides)
    return data


class TestAllPortsFull:
    """2 ports, 4 cars all arrive at t=1 — 2 served, 2 rejected."""

    def _data(self):
        return {
            "J": 2, "T": 3, "I": 4,
            "delta_t": 1.0,
            "P_max": 10_000.0,
            "M_big": 50.0,
            "epsilon": 0.05,
            "V":     {1: 400.0, 2: 400.0},
            "I_max": {1: 10.0,  2: 10.0},
            "p_buy":  {1: 0.10, 2: 0.10, 3: 0.10},
            "p_sell": {1: 0.20, 2: 0.20, 3: 0.20},
            "arr":    {1: 1, 2: 1, 3: 1, 4: 1},
            "t_max":  {1: 3, 2: 3, 3: 3, 4: 3},
            "s_init":   {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0},
            "s_target": {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
            "s_cap":    {1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0},
        }

    def test_rolling_horizon(self):
        data = self._data()
        res  = run_rolling_horizon(data, horizon=3)
        assert len(res.rejected) == 2
        assert len(res.assignments) == 2

    def test_equal_allocation(self):
        data = self._data()
        res  = run_equal_allocation(data)
        assert len(res.rejected) == 2
        assert len(res.assignments) == 2


class TestCarAlreadyAtTarget:
    """s_init == s_target — car should depart at the first step with zero energy charged."""

    def _data(self):
        return _base({"s_init": {1: 10.0}, "s_target": {1: 10.0}})

    def test_rolling_horizon_departs_immediately(self):
        data = self._data()
        res  = run_rolling_horizon(data, horizon=3)
        assert 1 in res.departed
        assert res.soc_final[1] >= data["s_target"][1] - 1e-2

    def test_equal_alloc_departs_immediately(self):
        data = self._data()
        res  = run_equal_allocation(data)
        assert 1 in res.departed
        assert res.soc_final[1] == pytest.approx(10.0, abs=1e-2)


class TestHorizonLongerThanRemaining:
    """H=10 with T=3 — build_rolling_model clamps to T, should not crash."""

    def test_no_crash_and_valid_result(self):
        data = _base({})
        res  = run_rolling_horizon(data, horizon=10)
        assert 1 in res.departed
        assert res.soc_final[1] >= data["s_target"][1] - 1e-2


class TestSingleTimestep:
    """T=1 — both runners must complete without error."""

    def _data(self):
        return {
            "J": 1, "T": 1, "I": 1,
            "delta_t": 1.0,
            "P_max": 5_000.0,
            "M_big": 50.0,
            "epsilon": 0.05,
            "V":     {1: 400.0},
            "I_max": {1: 10.0},
            "p_buy":  {1: 0.10},
            "p_sell": {1: 0.20},
            "arr":    {1: 1},
            "t_max":  {1: 1},
            "s_init":   {1: 5.0},
            "s_target": {1: 10.0},
            "s_cap":    {1: 20.0},
        }

    def test_rolling_horizon_completes(self):
        res = run_rolling_horizon(self._data(), horizon=1)
        assert isinstance(res.net_profit, float)

    def test_equal_alloc_completes(self):
        res = run_equal_allocation(self._data())
        assert isinstance(res.net_profit, float)


class TestCarArrivesAtLastStep:
    """arr[1]=T — car is assigned a port but has only one step to charge."""

    def _data(self):
        return _base({"arr": {1: 3}, "t_max": {1: 3}})

    def test_rolling_horizon_assigns_port(self):
        data = self._data()
        res  = run_rolling_horizon(data, horizon=1)
        assert 1 in res.assignments
        assert 1 not in res.rejected

    def test_equal_alloc_assigns_port(self):
        data = self._data()
        res  = run_equal_allocation(data)
        assert 1 in res.assignments
        assert 1 not in res.rejected


class TestNoCars:
    """I=0 — both runners must return empty results without error."""

    def _data(self):
        return {
            "J": 2, "T": 3, "I": 0,
            "delta_t": 1.0,
            "P_max": 5_000.0,
            "M_big": 50.0,
            "epsilon": 0.05,
            "V":     {1: 400.0, 2: 400.0},
            "I_max": {1: 10.0,  2: 10.0},
            "p_buy":  {1: 0.10, 2: 0.10, 3: 0.10},
            "p_sell": {1: 0.20, 2: 0.20, 3: 0.20},
            "arr": {}, "t_max": {}, "s_init": {}, "s_target": {}, "s_cap": {},
        }

    def test_rolling_horizon_empty(self):
        res = run_rolling_horizon(self._data(), horizon=3)
        assert res.soc_final     == {}
        assert res.assignments   == {}
        assert res.departed      == set()
        assert res.rejected      == set()
        assert res.net_profit    == pytest.approx(0.0)

    def test_equal_alloc_empty(self):
        res = run_equal_allocation(self._data())
        assert res.soc_final     == {}
        assert res.assignments   == {}
        assert res.departed      == set()
        assert res.rejected      == set()
        assert res.net_profit    == pytest.approx(0.0)
