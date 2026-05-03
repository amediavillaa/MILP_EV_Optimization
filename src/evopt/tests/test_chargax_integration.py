from evopt.experiments.station_configs import build_simple_station
from chargax import ChargingStation


def test_build_simple_station_returns_charging_station():
    station = build_simple_station(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    assert isinstance(station, ChargingStation)
