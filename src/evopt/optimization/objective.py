"""
objective.py — Objective function for EV charging MILP models
=============================================================
Both models maximise net profit:
    revenue  = sum over (i, j, t) of kWh delivered * p_sell[t]
    cost     = sum over (j, t)    of kWh bought    * p_buy[t]
    objective = revenue - cost

add_offline_objective(m)
    Uses Pyomo params (m.V, m.delta_t, m.p_sell, m.p_buy) and 3-indexed phi.

add_rolling_objective(m, data, assignments)
    Uses Python dicts (data, assignments) and 2-indexed phi, since port
    assignment is fixed and not stored as a Pyomo param in the rolling model.
"""

from pyomo.environ import Objective, maximize


def add_offline_objective(m) -> None:
    def obj_rule(m):
        revenue = sum(
            m.phi[i, j, t] * m.V[j] * (m.delta_t / 1000.0) * m.p_sell[t]
            for i in m.I for j in m.J for t in m.T
        )
        cost = sum(
            m.I_charge[j, t] * m.V[j] * (m.delta_t / 1000.0) * m.p_buy[t]
            for j in m.J for t in m.T
        )
        return revenue - cost

    m.obj = Objective(rule=obj_rule, sense=maximize)


def add_rolling_objective(m, data: dict, assignments: dict) -> None:
    cars = list(m.I)

    def obj_rule(m):
        revenue = sum(
            m.phi[i, t] * data["V"][assignments[i]] * (data["delta_t"] / 1000.0) * data["p_sell"][t]
            for i in cars for t in m.WIN
        )
        cost = sum(
            m.I_charge[j, t] * data["V"][j] * (data["delta_t"] / 1000.0) * data["p_buy"][t]
            for j in m.J for t in m.WIN
        )
        return revenue - cost

    m.obj = Objective(rule=obj_rule, sense=maximize)
