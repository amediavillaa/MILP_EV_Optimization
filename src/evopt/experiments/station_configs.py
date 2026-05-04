from chargax import EVSE, ChargingStation


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
