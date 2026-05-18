"""
objective.py — Profit-maximisation objective for EV charging LP models
=======================================================================
Profit maximisation:

    ev_revenue = Σ_{j,t} V_j · I_ev_{j,t} · z_{j,t} · p_eff_t · delta_t/1000
    grid_cost  = Σ_t (Σ_j V_j·I_ev_{j,t} + V_bess·I_bess_ch_t − V_bess·I_bess_dis_t)
                      · p_buy_t · delta_t/1000
    objective  = maximise (ev_revenue − grid_cost)

    p_eff_t = p_sell_t + ε·(t_last − t + 1)/H  where ε = 1e-4

    BESS discharge reduces the station's net grid draw rather than selling to the
    external grid.  It is therefore priced at the avoided buy cost (p_buy), not a
    separate sell price.  The arbitrage incentive is p_buy_expensive − p_buy_cheap:
    the LP charges the BESS when electricity is cheap and discharges when expensive.
"""

from pyomo.environ import Objective, minimize, value

_URGENCY_EPS = 1e-4


def add_offline_profit_objective(m, j_bess: int) -> None:
    def obj_rule(m):
        steps  = sorted(m.T)
        t_last = steps[-1]
        H      = len(steps)
        ev_revenue = sum(
            m.V[j] * m.I_ev[j, t] * value(m.z[j, t])
            * (m.p_sell[t] + _URGENCY_EPS * (t_last - t + 1) / H)
            * m.delta_t / 1000.0
            for j in m.J_ev for t in m.T
        )
        grid_cost = sum(
            (
                sum(m.V[j] * m.I_ev[j, t] for j in m.J_ev)
                + m.V[j_bess] * m.I_bess_ch[t]
                - m.V[j_bess] * m.I_bess_dis[t]
            )
            * m.p_buy[t] * m.delta_t / 1000.0
            for t in m.T
        )
        return -(ev_revenue - grid_cost)

    m.obj = Objective(rule=obj_rule, sense=minimize)


def add_rolling_profit_objective(m, j_bess: int) -> None:
    def obj_rule(m):
        steps  = sorted(m.WIN)
        t_last = steps[-1]
        H      = len(steps)
        ev_revenue = sum(
            m.V[j] * m.I_ev[j, t] * value(m.z[j, t])
            * (m.p_sell[t] + _URGENCY_EPS * (t_last - t + 1) / H)
            * m.delta_t / 1000.0
            for j in m.J_ev for t in m.WIN
        )
        grid_cost = sum(
            (
                sum(m.V[j] * m.I_ev[j, t] for j in m.J_ev)
                + m.V[j_bess] * m.I_bess_ch[t]
                - m.V[j_bess] * m.I_bess_dis[t]
            )
            * m.p_buy[t] * m.delta_t / 1000.0
            for t in m.WIN
        )
        return -(ev_revenue - grid_cost)

    m.obj = Objective(rule=obj_rule, sense=minimize)
