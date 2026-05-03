import pytest


@pytest.fixture
def single_car():
    """1 port, 1 car, 3 steps — hand-calculable ground truth."""
    return {
        "J": 1, "T": 3, "I": 1,
        "delta_t": 1.0,
        "P_max": 5_000.0,    # W — 5 kW, comfortably above one port's 4 kW max
        "M_big": 50.0,
        "epsilon": 0.05,
        "V":     {1: 400.0},
        "I_max": {1: 10.0},  # A → 4 kW max per port
        "p_buy":  {1: 0.10, 2: 0.10, 3: 0.10},
        "p_sell": {1: 0.20, 2: 0.20, 3: 0.20},
        "arr":    {1: 1},
        "t_max":  {1: 3},
        "s_init":   {1: 5.0},
        "s_target": {1: 10.0},  # needs 5 kWh; charges 4 kWh at t=1, 1 kWh at t=2 → departs
        "s_cap":    {1: 20.0},
    }


@pytest.fixture
def port_saturation():
    """1 port, 3 cars all arriving at t=1 — tests rejection and greedy assignment."""
    return {
        "J": 1, "T": 3, "I": 3,
        "delta_t": 1.0,
        "P_max": 5_000.0,
        "M_big": 50.0,
        "epsilon": 0.05,
        "V":     {1: 400.0},
        "I_max": {1: 10.0},
        "p_buy":  {1: 0.10, 2: 0.10, 3: 0.10},
        "p_sell": {1: 0.20, 2: 0.20, 3: 0.20},
        "arr":    {1: 1, 2: 1, 3: 1},
        "t_max":  {1: 3, 2: 3, 3: 3},
        "s_init":   {1: 5.0,  2: 5.0,  3: 5.0},
        "s_target": {1: 10.0, 2: 10.0, 3: 10.0},
        "s_cap":    {1: 20.0, 2: 20.0, 3: 20.0},
    }


@pytest.fixture
def sequential_arrival():
    """2 ports, 3 cars — car 1 departs at t=1, freeing port 1 for car 3 at t=2."""
    return {
        "J": 2, "T": 4, "I": 3,
        "delta_t": 1.0,
        "P_max": 10_000.0,   # W — no congestion; 2 ports × 4 kW = 8 kW < 10 kW
        "M_big": 50.0,
        "epsilon": 0.05,
        "V":     {1: 400.0, 2: 400.0},
        "I_max": {1: 10.0,  2: 10.0},
        "p_buy":  {1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10},
        "p_sell": {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20},
        "arr":    {1: 1, 2: 1, 3: 2},
        "t_max":  {1: 4, 2: 4, 3: 4},
        "s_init":   {1: 9.0,  2: 5.0,  3: 5.0},
        "s_target": {1: 10.0, 2: 15.0, 3: 10.0},
        "s_cap":    {1: 20.0, 2: 20.0, 3: 20.0},
    }
