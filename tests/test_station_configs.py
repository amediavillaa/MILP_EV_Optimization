from chargax import EVSE, ChargingStation, StationBattery

from milp_ev_opt.experiments.station_configs import build_station_with_battery


def test_returns_charging_station():
    st = build_station_with_battery(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    assert isinstance(st, ChargingStation)


def test_has_one_evse_and_one_battery():
    st = build_station_with_battery(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    evses = [c for c in st.connections if isinstance(c, EVSE)]
    batts = [c for c in st.connections if isinstance(c, StationBattery)]
    assert len(evses) == 1
    assert len(batts) == 1


def test_evse_has_correct_charger_count():
    st = build_station_with_battery(n_ports=2, v=400.0, i_max=32.0, p_max_kw=20.0)
    evse = next(c for c in st.connections if isinstance(c, EVSE))
    assert evse.num_chargers == 2


def test_battery_defaults():
    st = build_station_with_battery(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    batt = next(c for c in st.connections if isinstance(c, StationBattery))
    assert batt.capacity_kw == 30.0
    assert batt.max_kw_throughput == 10.0
    assert batt.efficiency == 0.95


def test_battery_custom_params():
    st = build_station_with_battery(
        n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0,
        batt_capacity_kwh=50.0, batt_max_kw=15.0, batt_efficiency=0.9,
    )
    batt = next(c for c in st.connections if isinstance(c, StationBattery))
    assert batt.capacity_kw == 50.0
    assert batt.max_kw_throughput == 15.0
    assert batt.efficiency == 0.9
