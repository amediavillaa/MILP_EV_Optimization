import types
import jax.numpy as jnp


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
