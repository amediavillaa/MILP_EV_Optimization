"""
variables.py — Variable declarations for EV charging MILP models
=================================================================
add_offline_variables(m)
    Adds all variables for the full offline MILP (build_ev_fcfs_model).
    Requires m.I, m.J, m.T to already be set on the model.

add_rolling_variables(m)
    Adds all variables for one MPC step (build_rolling_model).
    Requires m.I, m.J, m.WIN to already be set on the model.
    Port assignment (y) is absent — fixed externally on arrival.
    phi and x are 2-indexed (i, t) rather than 3-indexed (i, j, t)
    because each car has exactly one fixed port.
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
