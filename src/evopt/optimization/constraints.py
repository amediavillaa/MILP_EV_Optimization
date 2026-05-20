"""
constraints.py — Constraint sets for EV charging LP models
===========================================================
add_offline_constraints(m, data, j_bess, eta_bess)
    C1  EV port current upper bound
    C2  BESS current bounds (SoC-dependent ratios)
    C3  Grid power cap
    C5  BESS SoC dynamics
    C6  BESS SoC bounds
    C7  Car SoC dynamics
    C8  Car SoC bounds
    C9  Car charging power limit (SoC-dependent)
    C10 Occupancy enforcement

add_rolling_constraints(m, data, assignments, soc_now, socb_now, t_start, j_bess,
                        eta_bess)
    Same constraint set; window is m.WIN; warm-started from soc_now / socb_now.
"""

from pyomo.environ import Constraint, value

_ETA_BESS = 0.95   # one-way efficiency per direction; round-trip = 0.95^2 ~= 0.90


def add_offline_constraints(m, data: dict, j_bess: int,
                            eta_bess: float = _ETA_BESS) -> None:

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

    # C3 — Grid power cap (background load L reduces available headroom)
    def grid_cap_rule(m, t):
        ev_power   = sum(m.I_ev[j, t] * m.V[j] for j in m.J_ev)
        bess_power = (m.I_bess_ch[t] - m.I_bess_dis[t]) * m.V[j_bess]
        return ev_power + bess_power + m.L[t] <= m.P_max
    m.grid_cap = Constraint(m.T, rule=grid_cap_rule)

    # C5 — BESS SoC dynamics (raw commanded current, matching Chargax SoC update)
    # Eta appears only in the objective grid-cost terms, not here.
    def bess_soc_rule(m, t):
        net_in = (m.I_bess_ch[t] - m.I_bess_dis[t]) \
                 * m.V[j_bess] * (m.delta_t / 1000.0)
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
    eta_bess:    float = _ETA_BESS,
) -> None:

    # C1 — EV port current upper bound
    def ev_current_ub_rule(m, j, t):
        return m.I_ev[j, t] <= m.I_max[j]
    m.ev_current_ub = Constraint(m.J_ev, m.WIN, rule=ev_current_ub_rule)

    # C2 — BESS current bounds
    def bess_ch_ub_rule(m, t):
        return m.I_bess_ch[t] <= data["r_bess_ch"][t] * m.I_high
    def bess_dis_ub_rule(m, t):
        return m.I_bess_dis[t] <= data["r_bess_dis"][t] * m.I_low
    m.bess_ch_ub  = Constraint(m.WIN, rule=bess_ch_ub_rule)
    m.bess_dis_ub = Constraint(m.WIN, rule=bess_dis_ub_rule)

    # C3 — Grid power cap (background load L reduces available headroom)
    def grid_cap_rule(m, t):
        ev_power   = sum(m.I_ev[j, t] * m.V[j] for j in m.J_ev)
        bess_power = (m.I_bess_ch[t] - m.I_bess_dis[t]) * m.V[j_bess]
        return ev_power + bess_power + m.L[t] <= m.P_max
    m.grid_cap = Constraint(m.WIN, rule=grid_cap_rule)

    # C5 — BESS SoC dynamics (raw commanded current, warm-started from socb_now)
    def bess_soc_rule(m, t):
        net_in = (m.I_bess_ch[t] - m.I_bess_dis[t]) \
                 * m.V[j_bess] * (m.delta_t / 1000.0)
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

    # C8 — Car SoC bounds (cap at physical capacity s_cap, not s_target)
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


def add_must_serve_constraints(
    m,
    data:     dict,
    soc_now:  dict,   # {car_i: float}
    t_start:  int,
    t_end:    int,
) -> None:
    """C11 — Must-serve: enforce minimum pace NOW plus hard deadline within window.

    must_serve_pace: at every re-solve (t_start), each car must charge at least
        needed / steps_until_dep kWh this step.  With receding-horizon re-solves
        this accumulates exactly to the target at departure and prevents the LP
        from procrastinating (deferring all charging to the last step of the window
        and repeating that deferral on every re-solve).

    must_serve_dead: for cars whose departure falls inside the current window, a
        hard deadline forces full target by dep_i, independent of pacing.
    """
    cars = list(data["assignments"].keys())

    def _guard(i):
        """Return (needed, dep_i, steps_until_dep) or None if constraint should be skipped."""
        target = data["s_target"][i]
        needed = target - soc_now[i]
        if needed <= 1e-6:
            return None
        dep_i           = data["dep"][i]
        steps_until_dep = dep_i - t_start + 1
        if steps_until_dep <= 0:
            return None
        return needed, dep_i, steps_until_dep

    def must_serve_pace_rule(m, i):
        g = _guard(i)
        if g is None:
            return Constraint.Skip
        needed, dep_i, steps_until_dep = g
        j    = data["assignments"][i]
        V_j  = data["V"][j]
        dt   = data["delta_t"]
        n_cars = len(cars)

        # Pacing rate: energy per step needed to finish on time.
        # More urgent cars (short dwell, large deficit) get a higher floor.
        pacing_kwh = needed / steps_until_dep

        # Grid-share budget cap with 1% slack so the combined floor never
        # exactly saturates P_max — prevents floating-point infeasibility
        # that triggers bare-mode fallback and causes procrastination.
        grid_share_kwh = data["P_max"] * dt / (n_cars * 1000.0) * 0.99 if n_cars > 0 else 0.0

        # Port hardware cap: grid_share can exceed I_max at low occupancy.
        port_max_kwh = data["I_max"][j] * V_j * dt / 1000.0

        required_kwh = min(pacing_kwh, grid_share_kwh, port_max_kwh, needed)
        return m.soc_car[i, t_start] >= soc_now[i] + required_kwh

    def must_serve_dead_rule(m, i):
        g = _guard(i)
        if g is None:
            return Constraint.Skip
        needed, dep_i, steps_until_dep = g
        if dep_i > t_end:
            return Constraint.Skip
        # Only enforce deadline when target is physically achievable within dwell.
        max_energy_kwh = data["P_car_max"][i] * steps_until_dep * data["delta_t"] / 1000.0
        if soc_now[i] + max_energy_kwh < data["s_target"][i] - 1e-6:
            return Constraint.Skip
        return m.soc_car[i, dep_i] >= data["s_target"][i]

    m.must_serve_pace = Constraint(cars, rule=must_serve_pace_rule)
    m.must_serve_dead = Constraint(cars, rule=must_serve_dead_rule)
