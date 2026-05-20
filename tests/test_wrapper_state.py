import types
import pytest
import jax.numpy as jnp


def _make_evse_obs(connected: bool, soc_kw: float, capacity_kw: float,
                   desired_pct: float, time_till_leave: float) -> dict:
    evse = types.SimpleNamespace(
        charger_is_car_connected=jnp.array([connected]),
        car_battery_now_kw=jnp.array([soc_kw]),
        car_battery_capacity_kw=jnp.array([capacity_kw]),
        car_desired_battery_percentage=jnp.array([desired_pct]),
        car_time_till_leave=jnp.array([time_till_leave]),
    )
    return {
        "evses": evse,
        "future_buy_prices":  [0.20] * 24,
        "future_sell_prices": [0.15] * 24,
    }


def test_wrapper_extract_state_has_no_p_sell_bess():
    """extract_state should not include p_sell_bess in the returned state dict."""
    from evopt.env.chargax_wrapper import ChargaxWrapper

    wrapper = ChargaxWrapper(n_ports=1, v=400.0, i_max=32.0, p_max_kw=6.0)

    evse = types.SimpleNamespace(
        charger_is_car_connected=jnp.array([False]),
        car_battery_now_kw=jnp.array([0.0]),
        car_battery_capacity_kw=jnp.array([20.0]),
        car_desired_battery_percentage=jnp.array([0.8]),
        car_time_till_leave=jnp.array([60.0]),
    )
    obs = {
        "evses": evse,
        "future_buy_prices":  [0.20] * 24,
        "future_sell_prices": [0.15] * 24,
    }
    chargax_state = types.SimpleNamespace(timestep=0)

    state = wrapper.extract_state(obs, chargax_state)
    assert "p_sell_bess" not in state


def test_departure_fulfillment_below_target():
    """Car departing at soc=8kWh with target=10kWh should have fulfillment ratio 0.8."""
    from evopt.env.chargax_wrapper import ChargaxWrapper

    wrapper = ChargaxWrapper(n_ports=1, v=400.0, i_max=32.0, p_max_kw=6.0)

    # Step 0: car connects; soc=8, capacity=20, desired_pct=0.5 → target=10
    wrapper.extract_state(
        _make_evse_obs(True, 8.0, 20.0, 0.5, 60.0),
        types.SimpleNamespace(timestep=0),
    )

    # Step 1: car departs
    state = wrapper.extract_state(
        _make_evse_obs(False, 0.0, 20.0, 0.0, 0.0),
        types.SimpleNamespace(timestep=1),
    )

    assert state["departed_fulfillments"] == pytest.approx([0.8], rel=1e-6)


def test_departure_fulfillment_above_target_is_clamped():
    """Car departing at soc=12kWh with target=10kWh should have fulfillment ratio 1.0 (clamped)."""
    from evopt.env.chargax_wrapper import ChargaxWrapper

    wrapper = ChargaxWrapper(n_ports=1, v=400.0, i_max=32.0, p_max_kw=6.0)

    # Step 0: car connects; soc=12, capacity=20, desired_pct=0.5 → target=10
    wrapper.extract_state(
        _make_evse_obs(True, 12.0, 20.0, 0.5, 60.0),
        types.SimpleNamespace(timestep=0),
    )

    # Step 1: car departs
    state = wrapper.extract_state(
        _make_evse_obs(False, 0.0, 20.0, 0.0, 0.0),
        types.SimpleNamespace(timestep=1),
    )

    assert state["departed_fulfillments"] == pytest.approx([1.0], rel=1e-6)
