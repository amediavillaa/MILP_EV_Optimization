"""
Correctness tests — verify numerical outputs against hand-calculated values.

Expected values for single_car (1 port, 1 car, 3 steps, uniform prices):
  Step 1: charges 4 kWh (I_max=10A, V=400V, dt=1h) → SoC = 9.0 kWh
  Step 2: needs 1 kWh to reach target → SoC = 10.0 kWh → departs
  Revenue = (4 * 0.20) + (1 * 0.20) = €1.00
  Cost    = (4 * 0.10) + (1 * 0.10) = €0.50
  Net profit = €0.50
"""

import pytest
from evopt.experiments.runner import run_rolling_horizon, run_equal_allocation

TOL = 1e-2


class TestRollingHorizon:

    def test_single_car_final_soc(self, single_car):
        res = run_rolling_horizon(single_car, horizon=3)
        # MPC may charge beyond s_target (profitable when p_sell > p_buy); assert reached it
        assert res.soc_final[1] >= single_car["s_target"][1] - TOL

    def test_profit_identity(self, single_car):
        res = run_rolling_horizon(single_car, horizon=3)
        assert res.net_profit == pytest.approx(res.revenue - res.cost, abs=1e-6)

    def test_soc_trajectory_length(self, single_car):
        res = run_rolling_horizon(single_car, horizon=3)
        T = single_car["T"]
        assert len(res.soc_trajectory[1]) == T + 1

    def test_correct_port_assignment(self, port_saturation):
        res = run_rolling_horizon(port_saturation, horizon=3)
        assert res.assignments[1] == 1

    def test_rejected_cars_excluded(self, port_saturation):
        res = run_rolling_horizon(port_saturation, horizon=3)
        assert 1 not in res.rejected
        assert 2 in res.rejected
        assert 3 in res.rejected

    def test_car_departs_when_target_reached(self, single_car):
        res = run_rolling_horizon(single_car, horizon=3)
        assert 1 in res.departed


class TestEqualAllocation:

    def test_single_car_final_soc(self, single_car):
        res = run_equal_allocation(single_car)
        assert res.soc_final[1] == pytest.approx(single_car["s_target"][1], abs=TOL)

    def test_profit_identity(self, single_car):
        res = run_equal_allocation(single_car)
        assert res.net_profit == pytest.approx(res.revenue - res.cost, abs=1e-6)

    def test_soc_trajectory_length(self, single_car):
        res = run_equal_allocation(single_car)
        T = single_car["T"]
        assert len(res.soc_trajectory[1]) == T + 1

    def test_correct_port_assignment(self, port_saturation):
        res = run_equal_allocation(port_saturation)
        assert res.assignments[1] == 1

    def test_rejected_cars_excluded(self, port_saturation):
        res = run_equal_allocation(port_saturation)
        assert 1 not in res.rejected
        assert 2 in res.rejected
        assert 3 in res.rejected

    def test_car_departs_when_target_reached(self, single_car):
        res = run_equal_allocation(single_car)
        assert 1 in res.departed
