"""
objective.py — Objective functions for EV charging LP models
============================================================
Both models minimise a cost expression:
    charging_cost = Σ_{j,t} V_j · I_ev_{j,t} · z_{j,t} · (p_buy_t − p_sell_t)
    battery_cost  = Σ_t V_bess · (−I_bess_ch_t · p_buy_t + I_bess_dis_t · p_sell_t)
    objective     = charging_cost + battery_cost

add_offline_objective(m, j_bess)
    Uses Pyomo sets m.J_ev, m.T and params on m.

add_rolling_objective(m, j_bess)
    Same structure; uses m.WIN instead of m.T.
"""

from pyomo.environ import Objective, minimize, value


def add_offline_objective(m, j_bess: int) -> None:
    def obj_rule(m):
        charging_cost = sum(
            m.V[j] * m.I_ev[j, t] * value(m.z[j, t])
            * (m.p_buy[t] - m.p_sell[t])
            for j in m.J_ev for t in m.T
        )
        battery_cost = sum(
            m.V[j_bess] * (
                - m.I_bess_ch[t]  * m.p_buy[t]
                + m.I_bess_dis[t] * m.p_sell[t]
            )
            for t in m.T
        )
        return charging_cost + battery_cost

    m.obj = Objective(rule=obj_rule, sense=minimize)


def add_rolling_objective(m, j_bess: int) -> None:
    def obj_rule(m):
        charging_cost = sum(
            m.V[j] * m.I_ev[j, t] * value(m.z[j, t])
            * (m.p_buy[t] - m.p_sell[t])
            for j in m.J_ev for t in m.WIN
        )
        battery_cost = sum(
            m.V[j_bess] * (
                - m.I_bess_ch[t]  * m.p_buy[t]
                + m.I_bess_dis[t] * m.p_sell[t]
            )
            for t in m.WIN
        )
        return charging_cost + battery_cost

    m.obj = Objective(rule=obj_rule, sense=minimize)
