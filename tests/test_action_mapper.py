"""Tests for discretize_amps bidirectional mode."""
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


def test_rounding_uses_ceiling():
    # 17/32 * 10 = 5.3125 → ceil = 6 (at-or-above LP plan)
    assert discretize_amps(17.0, 32.0, 10) == 6


def test_zero_i_max_returns_zero():
    assert discretize_amps(10.0, 0.0, 10) == 0


def test_discretize_amps_bidirectional_full_charge():
    """I_max should map to level 2N in bidirectional mode."""
    assert discretize_amps(32.0, 32.0, 10, bidirectional=True) == 20


def test_discretize_amps_bidirectional_idle_at_zero():
    """0 amps should map to idle (N) in bidirectional mode."""
    assert discretize_amps(0.0, 32.0, 10, bidirectional=True) == 10


def test_discretize_amps_bidirectional_half_charge():
    """Half I_max should map to N + ceil(N/2) = 15 for N=10."""
    assert discretize_amps(16.0, 32.0, 10, bidirectional=True) == 15


def test_discretize_amps_unidirectional_full_charge():
    """I_max should map to level N in unidirectional mode."""
    assert discretize_amps(32.0, 32.0, 10, bidirectional=False) == 10


def test_discretize_amps_unidirectional_idle_at_zero():
    """0 amps should map to 0 in unidirectional mode."""
    assert discretize_amps(0.0, 32.0, 10, bidirectional=False) == 0
