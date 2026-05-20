"""
variables.py — Legacy MILP variable declarations (unused)
==========================================================
This module was part of an earlier FCFS-MILP formulation that modelled
port assignment (y), occupancy (x), and departure triggering (delta) as
binary decision variables.  The current formulation is a pure LP: port
assignments are precomputed by ChargaxWrapper, and none of the functions
here are imported or called.

Kept for reference only.
"""

from pyomo.environ import Var, NonNegativeReals, Binary


def add_offline_variables(m) -> None:
    m.I_charge = Var(m.J, m.T,      domain=NonNegativeReals)  # charging current (A)
    m.y        = Var(m.I, m.J,      domain=Binary)            # port assignment
    m.x        = Var(m.I, m.J, m.T, domain=Binary)            # occupancy
    m.delta    = Var(m.I, m.T,      domain=Binary)            # cumulative departure flag
    m.s        = Var(m.I, m.T,      domain=NonNegativeReals)  # state of charge (kWh)
    m.phi      = Var(m.I, m.J, m.T, domain=NonNegativeReals)  # McCormick aux


def add_rolling_variables(m) -> None:
    m.I_charge = Var(m.J, m.WIN, domain=NonNegativeReals)  # charging current (A)
    m.x        = Var(m.I, m.WIN, domain=Binary)            # occupancy
    m.delta    = Var(m.I, m.WIN, domain=Binary)            # cumulative departure flag
    m.s        = Var(m.I, m.WIN, domain=NonNegativeReals)  # state of charge (kWh)
    m.phi      = Var(m.I, m.WIN, domain=NonNegativeReals)  # McCormick aux
