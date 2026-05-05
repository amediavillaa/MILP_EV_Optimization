from chargax import EVSE, ChargingStation, StationBattery


def build_simple_station(
    n_ports: int,
    v: float,
    i_max: float,
    p_max_kw: float,
) -> ChargingStation:
    """Simple station with identical unidirectional ports and no battery."""
    evse = EVSE(
        num_chargers=n_ports,
        voltage=v,
        max_current=i_max,
        efficiency=1.0,
    )
    return ChargingStation(
        max_kw_throughput=p_max_kw,
        efficiency=1.0,
        connections=[evse],
    )


def build_station_with_battery(
    n_ports: int,
    v: float,
    i_max: float,
    p_max_kw: float,
    batt_capacity_kwh: float = 30.0,
    batt_max_kw: float = 10.0,
    batt_efficiency: float = 0.95,
) -> ChargingStation:
    """Station with n_ports EV chargers and one on-site BESS.

    Note: StationBattery.capacity_kw stores the energy capacity in kWh despite
    the field name using the 'kw' suffix — this is a known Chargax quirk.
    """
    evse = EVSE(num_chargers=n_ports, voltage=v, max_current=i_max, efficiency=1.0)
    batt = StationBattery(
        capacity_kw=batt_capacity_kwh,
        max_kw_throughput=batt_max_kw,
        efficiency=batt_efficiency,
    )
    return ChargingStation(
        max_kw_throughput=p_max_kw,
        efficiency=1.0,
        connections=[evse, batt],
    )
