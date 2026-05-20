"""Tests for discretize_amps bidirectional mode."""
import pytest


def test_discretize_amps_bidirectional_full_charge():
    """I_max should map to level 2N in bidirectional mode."""
    from evopt.env.action_mapper import discretize_amps
    assert discretize_amps(32.0, 32.0, 10, bidirectional=True) == 20


def test_discretize_amps_bidirectional_idle_at_zero():
    """0 amps should map to idle (N) in bidirectional mode."""
    from evopt.env.action_mapper import discretize_amps
    assert discretize_amps(0.0, 32.0, 10, bidirectional=True) == 10


def test_discretize_amps_bidirectional_half_charge():
    """Half I_max should map to N + ceil(N/2) = 15 for N=10."""
    from evopt.env.action_mapper import discretize_amps
    assert discretize_amps(16.0, 32.0, 10, bidirectional=True) == 15


def test_discretize_amps_unidirectional_full_charge():
    """I_max should map to level N in unidirectional mode."""
    from evopt.env.action_mapper import discretize_amps
    assert discretize_amps(32.0, 32.0, 10, bidirectional=False) == 10


def test_discretize_amps_unidirectional_idle_at_zero():
    """0 amps should map to 0 in unidirectional mode."""
    from evopt.env.action_mapper import discretize_amps
    assert discretize_amps(0.0, 32.0, 10, bidirectional=False) == 0
