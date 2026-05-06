"""
constraints.py — Constraint sets for EV charging LP models
===========================================================
add_offline_constraints(m, data, j_bess)
    C1  EV port current upper bound
    C2  BESS current bounds (SoC-dependent ratios)
    C3  Grid power cap
    C5  BESS SoC dynamics
    C6  BESS SoC bounds
    C7  Car SoC dynamics
    C8  Car SoC bounds
    C9  Car charging power limit (SoC-dependent)
    C10 Occupancy enforcement

add_rolling_constraints(m, data, assignments, soc_now, socb_now, t_start, j_bess)
    Same constraint set; window is m.WIN; warm-started from soc_now / socb_now.
"""

from pyomo.environ import Constraint, value


def add_offline_constraints(m, data: dict, j_bess: int) -> None:

    # C1 — EV port current upper bound
    def ev_current_ub_rule(m, j, t):
        return m.I_ev[j, t] <= m.I_max[j]
    m.ev_current_ub = Constraint(m.J_ev, m.T, rule=ev_current_ub_rule)

    # C2 — BESS current bounds (SoC-dependent ratios)
    def bess_ch_ub_rule(m, t):
        return m.I_bess_ch[t] <= data["r_bess_ch"][t] * m.I_high
    def bess_dis_ub_rule(m, t):
        return m.I_bess_dis[t] <= data["r_bess_dis"][t] * m.I_low
    m.bess_ch_ub  = Constraint(m.T, rule=bess_ch_ub_rule)
    m.bess_dis_ub = Constraint(m.T, rule=bess_dis_ub_rule)

    # C3 — Grid power cap
    def grid_cap_rule(m, t):
        ev_power   = sum(m.I_ev[j, t] * m.V[j] for j in m.J_ev)
        bess_power = (m.I_bess_ch[t] - m.I_bess_dis[t]) * m.V[j_bess]
        return ev_power + bess_power <= m.P_max
    m.grid_cap = Constraint(m.T, rule=grid_cap_rule)

    # C5 — BESS SoC dynamics
    def bess_soc_rule(m, t):
        net_in = (m.I_bess_ch[t] - m.I_bess_dis[t]) * m.V[j_bess] \
                 * (m.delta_t / 1000.0)
        if t == 1:
            return m.SoCB[t] == m.SoCB_init + net_in
        return m.SoCB[t] == m.SoCB[t - 1] + net_in
    m.bess_soc = Constraint(m.T, rule=bess_soc_rule)

    # C6 — BESS SoC bounds
    def bess_soc_lb_rule(m, t):
        return m.SoCB[t] >= m.SoCB_min
    def bess_soc_ub_rule(m, t):
        return m.SoCB[t] <= m.SoCB_max
    m.bess_soc_lb = Constraint(m.T, rule=bess_soc_lb_rule)
    m.bess_soc_ub = Constraint(m.T, rule=bess_soc_ub_rule)

    # C7 — Car SoC dynamics
    def car_soc_rule(m, i, t):
        j         = data["assignments"][i]
        energy_in = m.I_ev[j, t] * m.V[j] * (m.delta_t / 1000.0)
        if t < data["arr"][i]:
            return m.soc_car[i, t] == data["s_init"][i]
        elif t == data["arr"][i]:
            return m.soc_car[i, t] == data["s_init"][i] + energy_in
        elif t <= data["dep"][i]:
            return m.soc_car[i, t] == m.soc_car[i, t - 1] + energy_in
        else:
            return m.soc_car[i, t] == m.soc_car[i, t - 1]
    m.car_soc = Constraint(m.I, m.T, rule=car_soc_rule)

    # C8 — Car SoC bounds
    def car_soc_lb_rule(m, i, t):
        return m.soc_car[i, t] >= data["s_min"][i]
    def car_soc_ub_rule(m, i, t):
        return m.soc_car[i, t] <= data["s_cap"][i]
    m.car_soc_lb = Constraint(m.I, m.T, rule=car_soc_lb_rule)
    m.car_soc_ub = Constraint(m.I, m.T, rule=car_soc_ub_rule)

    # C9 — Car charging power limit (SoC-dependent)
    def car_power_limit_rule(m, i, t):
        if t < data["arr"][i] or t > data["dep"][i]:
            return Constraint.Skip
        j = data["assignments"][i]
        return m.I_ev[j, t] * m.V[j] <= data["r_car"][i, t] * data["P_car_max"][i]
    m.car_power_limit = Constraint(m.I, m.T, rule=car_power_limit_rule)

    # C10 — Occupancy enforcement: no current on empty ports
    def occupancy_rule(m, j, t):
        if value(m.z[j, t]) == 0:
            return m.I_ev[j, t] == 0
        return Constraint.Skip
    m.occupancy = Constraint(m.J_ev, m.T, rule=occupancy_rule)


def add_rolling_constraints(
    m,
    data:        dict,
    assignments: dict,   # {car_i: port_j}
    soc_now:     dict,   # {car_i: float}
    socb_now:    float,
    t_start:     int,
    j_bess:      int,
) -> None:

    # C1 — EV port current upper bound
    def ev_current_ub_rule(m, j, t):
        return m.I_ev[j, t] <= m.I_max[j]
    m.ev_current_ub = Constraint(m.J_ev, m.WIN, rule=ev_current_ub_rule)

    # C2 — BESS current bounds
    def bess_ch_ub_rule(m, t):
        return m.I_bess_ch[t] <= data["r_bess_ch"][t] * value(m.I_high)
    def bess_dis_ub_rule(m, t):
        return m.I_bess_dis[t] <= data["r_bess_dis"][t] * value(m.I_low)
    m.bess_ch_ub  = Constraint(m.WIN, rule=bess_ch_ub_rule)
    m.bess_dis_ub = Constraint(m.WIN, rule=bess_dis_ub_rule)

    # C3 — Grid power cap
    def grid_cap_rule(m, t):
        ev_power   = sum(m.I_ev[j, t] * m.V[j] for j in m.J_ev)
        bess_power = (m.I_bess_ch[t] - m.I_bess_dis[t]) * m.V[j_bess]
        return ev_power + bess_power <= m.P_max
    m.grid_cap = Constraint(m.WIN, rule=grid_cap_rule)

    # C5 — BESS SoC dynamics (warm-started from socb_now)
    def bess_soc_rule(m, t):
        net_in = (m.I_bess_ch[t] - m.I_bess_dis[t]) * m.V[j_bess] \
                 * (m.delta_t / 1000.0)
        if t == t_start:
            return m.SoCB[t] == socb_now + net_in
        return m.SoCB[t] == m.SoCB[t - 1] + net_in
    m.bess_soc = Constraint(m.WIN, rule=bess_soc_rule)

    # C6 — BESS SoC bounds
    def bess_soc_lb_rule(m, t):
        return m.SoCB[t] >= m.SoCB_min
    def bess_soc_ub_rule(m, t):
        return m.SoCB[t] <= m.SoCB_max
    m.bess_soc_lb = Constraint(m.WIN, rule=bess_soc_lb_rule)
    m.bess_soc_ub = Constraint(m.WIN, rule=bess_soc_ub_rule)

    # C7 — Car SoC dynamics (warm-started from soc_now)
    def car_soc_rule(m, i, t):
        j         = assignments[i]
        energy_in = m.I_ev[j, t] * m.V[j] * (m.delta_t / 1000.0)
        if t == t_start:
            return m.soc_car[i, t] == soc_now[i] + energy_in
        if t <= data["dep"][i]:
            return m.soc_car[i, t] == m.soc_car[i, t - 1] + energy_in
        return m.soc_car[i, t] == m.soc_car[i, t - 1]
    m.car_soc = Constraint(m.I, m.WIN, rule=car_soc_rule)

    # C8 — Car SoC bounds
    def car_soc_lb_rule(m, i, t):
        return m.soc_car[i, t] >= m.s_min[i]
    def car_soc_ub_rule(m, i, t):
        return m.soc_car[i, t] <= m.s_cap[i]
    m.car_soc_lb = Constraint(m.I, m.WIN, rule=car_soc_lb_rule)
    m.car_soc_ub = Constraint(m.I, m.WIN, rule=car_soc_ub_rule)

    # C9 — Car charging power limit (SoC-dependent)
    def car_power_limit_rule(m, i, t):
        if t > data["dep"][i]:
            return Constraint.Skip
        j = assignments[i]
        return m.I_ev[j, t] * m.V[j] <= data["r_car"][i, t] * data["P_car_max"][i]
    m.car_power_limit = Constraint(m.I, m.WIN, rule=car_power_limit_rule)

    # C10 — Occupancy enforcement
    def occupancy_rule(m, j, t):
        if value(m.z[j, t]) == 0:
            return m.I_ev[j, t] == 0
        return Constraint.Skip
    m.occupancy = Constraint(m.J_ev, m.WIN, rule=occupancy_rule)
