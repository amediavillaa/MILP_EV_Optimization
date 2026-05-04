import pytest
from evopt.env.chargax_wrapper import ChargaxWrapper


class MockEVSE:
    def __init__(self, n, connected, soc, capacity, desired_pct,
                 time_till_leave, voltage=400.0, max_current=32.0):
        self.charger_is_car_connected = connected
        self.car_battery_now_kw       = soc
        self.car_battery_capacity_kw  = capacity
        self.car_desired_battery_percentage = desired_pct
        self.car_time_till_leave      = time_till_leave
        self.voltage                  = voltage
        self.max_current              = max_current


class MockState:
    def __init__(self, timestep=0, profit=0.0, served=0, rejected=0, dt="2023-01-01",
                 customer_price=0.75):
        self.timestep                  = timestep
        self.profit                    = profit
        self.served_customers          = served
        self.rejected_customers        = rejected
        self.datetime                  = dt
        self.elec_customer_sell_price  = customer_price


def _make_obs(evse, buy_prices=None, sell_prices=None):
    return {
        "evses": evse,
        "future_buy_prices":  buy_prices  or [0.20] * 6,
        "future_sell_prices": sell_prices or [0.40] * 6,
    }


def _make_wrapper(n_ports=2):
    return ChargaxWrapper(
        n_ports=n_ports,
        v=400.0,
        i_max=32.0,
        p_max_kw=20.0,
        num_discretization_levels=10,
        minutes_per_step=5,
    )


def test_reset_clears_state():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [True, False], [20.0, 0.0], [60.0, 60.0],
                    [0.8, 0.0], [60.0, 0.0])
    w.extract_state(_make_obs(evse), MockState(0))
    w.reset()
    assert w._charger_to_car == {}
    assert w._prev_connected == set()
    assert w._next_car_id == 0


def test_no_cars_returns_empty_present():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [False, False], [0.0, 0.0], [60.0, 60.0],
                    [0.8, 0.8], [60.0, 60.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert state["present_cars"] == {}
    assert state["assignments"] == {}


def test_arrival_assigns_new_car_id():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [True, False], [20.0, 0.0], [60.0, 60.0],
                    [0.8, 0.0], [60.0, 0.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert len(state["present_cars"]) == 1
    car_id = list(state["present_cars"].keys())[0]
    assert state["assignments"][car_id] == 1  # charger 0 → port 1


def test_two_arrivals_get_distinct_ids():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [True, True], [20.0, 15.0], [60.0, 60.0],
                    [0.8, 0.8], [60.0, 60.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert len(state["present_cars"]) == 2
    ids = list(state["present_cars"].keys())
    assert ids[0] != ids[1]


def test_departure_removes_car():
    w = _make_wrapper(2)
    evse_on = MockEVSE(2, [True, False], [20.0, 0.0], [60.0, 60.0],
                       [0.8, 0.0], [60.0, 0.0])
    state1 = w.extract_state(_make_obs(evse_on), MockState(0))
    car_id = list(state1["present_cars"].keys())[0]

    evse_off = MockEVSE(2, [False, False], [0.0, 0.0], [60.0, 60.0],
                        [0.0, 0.0], [0.0, 0.0])
    state2 = w.extract_state(_make_obs(evse_off), MockState(1))
    assert car_id not in state2["present_cars"]
    assert state2["departed_socs"] == [pytest.approx(20.0)]


def test_soc_fields_extracted_correctly():
    w = _make_wrapper(1)
    evse = MockEVSE(1, [True], [25.0], [60.0], [0.9], [30.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    car = list(state["present_cars"].values())[0]
    assert car["soc_now"]  == pytest.approx(25.0)
    assert car["s_cap"]    == pytest.approx(60.0)
    assert car["s_target"] == pytest.approx(54.0)  # 0.9 * 60


def test_p_max_converted_to_watts():
    w = _make_wrapper(1)
    evse = MockEVSE(1, [False], [0.0], [60.0], [0.0], [0.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert state["P_max"] == pytest.approx(20_000.0)  # 20 kW → 20000 W


def test_price_dict_populated():
    w = _make_wrapper(1)
    evse = MockEVSE(1, [False], [0.0], [60.0], [0.0], [0.0])
    state = w.extract_state(_make_obs(evse, buy_prices=[0.20] * 6),
                            MockState(0, customer_price=0.75))
    assert state["p_buy"][0] == pytest.approx(0.20)
    assert state["p_sell"][0] == pytest.approx(0.75)  # customer fee, not V2G price
    assert 0 in state["p_buy"]


def test_to_chargax_actions_full_charge():
    w = _make_wrapper(2)
    actions = w.to_chargax_actions({1: 32.0, 2: 16.0})
    # evses is a list (one array per EVSE group); index into the first group's array
    evse_arr = list(actions["evses"][0])
    assert evse_arr[0] == 10   # 32/32 * 10 = 10
    assert evse_arr[1] == 5    # 16/32 * 10 = 5


def test_to_chargax_actions_missing_port_defaults_to_zero():
    w = _make_wrapper(2)
    actions = w.to_chargax_actions({1: 32.0})  # port 2 missing
    evse_arr = list(actions["evses"][0])
    assert evse_arr[1] == 0
