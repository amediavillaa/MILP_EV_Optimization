import pytest


@pytest.fixture
def minimal_bess_data():
    """1-port, 1-car, 2-step offline LP scenario with a pre-charged BESS."""
    J = 1
    j_bess = J + 1
    return {
        "J": J,
        "T": 2,
        "I": 1,
        "delta_t": 1.0,            # 1 hour per step — simplifies energy arithmetic
        "P_max": 20_000.0,         # 20 kW — non-binding
        "V":      {1: 400.0, j_bess: 400.0},
        "I_max":  {1: 32.0},
        "I_high": 25.0,
        "I_low":  25.0,
        "p_buy":  {1: 0.10, 2: 0.30},   # cheap at t=1, expensive at t=2
        "p_sell": {1: 0.50, 2: 0.50},
        "L":      {1: 0.0,  2: 0.0},
        "SoCB_min":  0.0,
        "SoCB_max":  20.0,
        "SoCB_init": 10.0,         # BESS starts half-charged
        "r_bess_ch":  {1: 1.0, 2: 1.0},
        "r_bess_dis": {1: 1.0, 2: 1.0},
        "assignments": {1: 1},
        "arr": {1: 1},
        "dep": {1: 2},
        "s_init":    {1: 0.0},
        "s_cap":     {1: 20.0},
        "s_min":     {1: 0.0},
        "s_target":  {1: 10.0},
        "P_car_max": {1: 12_800.0},    # 400 V × 32 A
        "r_car":     {(1, 1): 1.0, (1, 2): 1.0},
    }
