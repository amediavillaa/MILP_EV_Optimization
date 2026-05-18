import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock


def _make_mock_runner(bess_net_amps: float, p_buy: float = 0.30,
                      delta_t: float = 5 / 60.0):
    """Build a mocked BenchmarkRunner for a single-step episode with no EVs."""
    from evopt.benchmarking.runner import BenchmarkRunner

    J = 2
    t = 0

    wrapper = MagicMock()
    wrapper.v_bess = 400.0

    clean_state = {
        "t": t,
        "delta_t": delta_t,
        "J": J,
        "P_max": 20_000.0,
        "V": {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0, 2: 32.0},
        "p_buy":  {t: p_buy},
        "p_sell": {t: 0.50},
        "present_cars": {},
        "assignments": {},
        "departed_socs": [],
    }
    wrapper.extract_state.return_value = clean_state
    wrapper.to_chargax_actions.return_value = {
        "evses": [jnp.array([5, 5], dtype=jnp.int32)],
        "batteries": [],
    }

    controller = MagicMock()
    controller.compute_action.return_value = {J + 1: bess_net_amps}

    mock_state = MagicMock()
    mock_state.profit = 0.0
    mock_state.served_customers = 0
    mock_state.rejected_customers = 0
    mock_state.datetime = "2024-01-01"

    mock_timestep = MagicMock()
    mock_timestep.reward = 0.0
    mock_timestep.observation = {}
    mock_timestep.terminated = jnp.array(True)
    mock_timestep.truncated = jnp.array(False)

    env = MagicMock()
    env.reset_env.return_value = ({}, mock_state)
    env.step_env.return_value = (mock_timestep, mock_state)

    return BenchmarkRunner(env, wrapper), controller


def test_runner_bess_charging_adds_to_step_cost():
    """BESS charging (bess_net < 0) should add energy × p_buy to step_cost."""
    delta_t = 5 / 60.0
    p_buy = 0.30
    # LP convention: bess_net = I_dis − I_ch → charging 25 A gives bess_net = −25
    runner, ctrl = _make_mock_runner(bess_net_amps=-25.0, p_buy=p_buy, delta_t=delta_t)
    result = runner.run_episode(ctrl, seed=0)

    expected = 25.0 * 400.0 * delta_t / 1000.0 * p_buy
    assert result.step_log[0].step_cost == pytest.approx(expected, rel=1e-3)


def test_runner_bess_discharging_reduces_step_cost():
    """BESS discharging (bess_net > 0) should subtract energy × p_buy from step_cost."""
    delta_t = 5 / 60.0
    p_buy = 0.30
    runner, ctrl = _make_mock_runner(bess_net_amps=25.0, p_buy=p_buy, delta_t=delta_t)
    result = runner.run_episode(ctrl, seed=0)

    expected = -25.0 * 400.0 * delta_t / 1000.0 * p_buy
    assert result.step_log[0].step_cost == pytest.approx(expected, rel=1e-3)


def test_runner_no_bess_step_cost_unaffected():
    """When wrapper.v_bess is None, BESS actions should not change step_cost."""
    from evopt.benchmarking.runner import BenchmarkRunner

    J = 2
    t = 0
    delta_t = 5 / 60.0

    wrapper = MagicMock()
    wrapper.v_bess = None

    clean_state = {
        "t": t,
        "delta_t": delta_t,
        "J": J,
        "P_max": 20_000.0,
        "V": {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0, 2: 32.0},
        "p_buy":  {t: 0.30},
        "p_sell": {t: 0.50},
        "present_cars": {},
        "assignments": {},
        "departed_socs": [],
    }
    wrapper.extract_state.return_value = clean_state
    wrapper.to_chargax_actions.return_value = {
        "evses": [jnp.array([5, 5], dtype=jnp.int32)],
        "batteries": [],
    }

    controller = MagicMock()
    controller.compute_action.return_value = {J + 1: -25.0}

    mock_state = MagicMock()
    mock_state.profit = 0.0
    mock_state.served_customers = 0
    mock_state.rejected_customers = 0
    mock_state.datetime = "2024-01-01"

    mock_timestep = MagicMock()
    mock_timestep.reward = 0.0
    mock_timestep.observation = {}
    mock_timestep.terminated = jnp.array(True)
    mock_timestep.truncated = jnp.array(False)

    env = MagicMock()
    env.reset_env.return_value = ({}, mock_state)
    env.step_env.return_value = (mock_timestep, mock_state)

    runner = BenchmarkRunner(env, wrapper)
    result = runner.run_episode(controller, seed=0)

    assert result.step_log[0].step_cost == pytest.approx(0.0, abs=1e-9)
