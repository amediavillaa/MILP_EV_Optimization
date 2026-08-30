def test_lp_data_has_no_p_sell_bess():
    """_build_lp_data should not include p_sell_bess in the output dict."""
    from milp_ev_opt.controllers.lp_controller import LPController

    ctrl = LPController(horizon_steps=2)
    state = {
        "t": 1,
        "J": 1,
        "delta_t": 1 / 12,
        "P_max": 6_000.0,
        "V":     {1: 400.0},
        "I_max": {1: 32.0},
        "p_buy":       {t: 0.20 for t in range(1, 300)},
        "p_sell":      {t: 0.50 for t in range(1, 300)},
        "p_sell_bess": {t: 0.15 for t in range(1, 300)},
        "present_cars": {
            1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 12},
        },
        "assignments": {1: 1},
    }
    data = ctrl._build_lp_data(state)
    assert "p_sell_bess" not in data
