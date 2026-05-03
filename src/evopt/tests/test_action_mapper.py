import pytest
from evopt.env.action_mapper import discretize_amps


def test_full_current_maps_to_max_level():
    assert discretize_amps(32.0, 32.0, 10) == 10


def test_zero_current_maps_to_zero():
    assert discretize_amps(0.0, 32.0, 10) == 0


def test_half_current_maps_to_half_level():
    assert discretize_amps(16.0, 32.0, 10) == 5


def test_overcurrent_clamps_to_max_level():
    assert discretize_amps(40.0, 32.0, 10) == 10


def test_negative_current_clamps_to_zero():
    assert discretize_amps(-5.0, 32.0, 10) == 0


def test_rounding_rounds_to_nearest():
    # 17/32 * 10 = 5.3125 → rounds to 5
    assert discretize_amps(17.0, 32.0, 10) == 5


def test_zero_i_max_returns_zero():
    assert discretize_amps(10.0, 0.0, 10) == 0
