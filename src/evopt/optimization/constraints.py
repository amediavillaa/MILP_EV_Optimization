"""
constraints.py — Constraint sets for EV charging MILP models
=============================================================
add_offline_constraints(m)
    Adds all constraints for the full offline MILP.
    Reads exclusively from Pyomo params on m — no extra arguments needed.

    C1  SoC dynamics
    C2  McCormick linearisation  phi[i,j,t] = x[i,j,t] * I_charge[j,t]
    C3  Departure condition      delta flips when s >= s_target
    C4  Occupancy                linked to assignment y and departure delta
    C5  Port capacity            at most one car per port per step
    C6  FCFS assignment          y variables + must-assign-if-free
    C7  Current → occupied ports I_charge = 0 on empty ports
    C8  Grid power cap
    C9  SoC bounds

add_rolling_constraints(m, data, assignments, soc_now, t_start)
    Adds all constraints for one rolling MPC step.
    Uses Python dicts (data, assignments, soc_now) because port assignment
    is fixed externally and not stored as a Pyomo param.
    Omits C6 (no assignment decisions) and C9's pre-arrival guard
    (all cars are already present at t_start).

    C1  SoC dynamics
    C2  McCormick linearisation  phi[i,t] = x[i,t] * I_charge[assign[i],t]
    C3  Departure condition
    C4  Occupancy
    C5  Port capacity
    C6  Current → occupied ports
    C7  Grid power cap
    C8  SoC bounds
"""

from pyomo.environ import Constraint, value


# ---------------------------------------------------------------------------
# Offline constraints
# ---------------------------------------------------------------------------

def add_offline_constraints(m) -> None:

    # ------------------------------------------------------------------
    # C1 — SoC dynamics
    # ------------------------------------------------------------------
    def soc_dynamics_rule(m, i, t):
        energy_in = sum(
            m.phi[i, j, t] * m.V[j] * (m.delta_t / 1000.0)
            for j in m.J
        )
        if t == value(m.arr[i]):
            return m.s[i, t] == m.s_init[i] + energy_in
        elif t > value(m.arr[i]):
            return m.s[i, t] == m.s[i, t - 1] + energy_in
        else:
            return m.s[i, t] == m.s_init[i]

    m.soc_dynamics = Constraint(m.I, m.T, rule=soc_dynamics_rule)

    # ------------------------------------------------------------------
    # C2 — McCormick linearisation: phi[i,j,t] = x[i,j,t] * I_charge[j,t]
    # ------------------------------------------------------------------
    def phi_ub_x_rule(m, i, j, t):
        return m.phi[i, j, t] <= m.I_max[j] * m.x[i, j, t]

    def phi_ub_I_rule(m, i, j, t):
        return m.phi[i, j, t] <= m.I_charge[j, t]

    def phi_lb_rule(m, i, j, t):
        return m.phi[i, j, t] >= m.I_charge[j, t] - m.I_max[j] * (1 - m.x[i, j, t])

    m.phi_ub_x = Constraint(m.I, m.J, m.T, rule=phi_ub_x_rule)
    m.phi_ub_I = Constraint(m.I, m.J, m.T, rule=phi_ub_I_rule)
    m.phi_lb   = Constraint(m.I, m.J, m.T, rule=phi_lb_rule)

    # ------------------------------------------------------------------
    # C3 — Departure condition
    # ------------------------------------------------------------------
    def depart_lb_rule(m, i, t):
        return m.s[i, t] >= m.s_target[i] * m.delta[i, t]

    def depart_ub_rule(m, i, t):
        return m.s[i, t] <= m.s_target[i] - m.epsilon + m.M_big * m.delta[i, t]

    def depart_monotone_rule(m, i, t):
        if t == 1:
            return Constraint.Skip
        return m.delta[i, t] >= m.delta[i, t - 1]

    def depart_before_arrival_rule(m, i, t):
        if t < value(m.arr[i]):
            return m.delta[i, t] == 0
        return Constraint.Skip

    m.depart_lb         = Constraint(m.I, m.T, rule=depart_lb_rule)
    m.depart_ub         = Constraint(m.I, m.T, rule=depart_ub_rule)
    m.depart_monotone   = Constraint(m.I, m.T, rule=depart_monotone_rule)
    m.depart_before_arr = Constraint(m.I, m.T, rule=depart_before_arrival_rule)

    # ------------------------------------------------------------------
    # C4 — Occupancy linked to assignment and departure
    # ------------------------------------------------------------------
    def occ_assignment_rule(m, i, j, t):
        return m.x[i, j, t] <= m.y[i, j]

    def occ_after_depart_rule(m, i, j, t):
        if t == 1:
            return Constraint.Skip
        return m.x[i, j, t] <= 1 - m.delta[i, t - 1]

    def occ_before_arrival_rule(m, i, j, t):
        if t < value(m.arr[i]):
            return m.x[i, j, t] == 0
        return Constraint.Skip

    def occ_presence_rule(m, i, t):
        if t < value(m.arr[i]):
            return Constraint.Skip
        assigned      = sum(m.y[i, j] for j in m.J)
        present       = sum(m.x[i, j, t] for j in m.J)
        departed_prev = 0 if t == 1 else m.delta[i, t - 1]
        return present >= assigned - departed_prev

    m.occ_assignment   = Constraint(m.I, m.J, m.T, rule=occ_assignment_rule)
    m.occ_after_depart = Constraint(m.I, m.J, m.T, rule=occ_after_depart_rule)
    m.occ_before_arr   = Constraint(m.I, m.J, m.T, rule=occ_before_arrival_rule)
    m.occ_presence     = Constraint(m.I, m.T,      rule=occ_presence_rule)

    # ------------------------------------------------------------------
    # C5 — Port capacity: at most one car per port per time step
    # ------------------------------------------------------------------
    def port_capacity_rule(m, j, t):
        return sum(m.x[i, j, t] for i in m.I) <= 1

    m.port_capacity = Constraint(m.J, m.T, rule=port_capacity_rule)

    # ------------------------------------------------------------------
    # C6 — FCFS assignment
    # ------------------------------------------------------------------
    def one_port_rule(m, i):
        return sum(m.y[i, j] for j in m.J) <= 1

    def fcfs_rule(m, i, ip, j):
        t_before_ip = value(m.arr[ip]) - 1
        if t_before_ip < 1:
            return Constraint.Skip
        return m.y[ip, j] <= m.delta[i, t_before_ip] + (1 - m.y[i, j])

    def must_assign_if_free_rule(m, i):
        t_arr   = value(m.arr[i])
        n_ports = len(list(m.J))
        occupied = sum(
            m.x[ip, j, t_arr]
            for ip in m.I if ip != i
            for j in m.J
        )
        return n_ports * sum(m.y[i, j] for j in m.J) + occupied >= n_ports

    m.one_port            = Constraint(m.I,              rule=one_port_rule)
    m.fcfs                = Constraint(m.FCFS_pairs, m.J, rule=fcfs_rule)
    m.must_assign_if_free = Constraint(m.I,              rule=must_assign_if_free_rule)

    # ------------------------------------------------------------------
    # C7 — Current only flows to occupied ports
    # ------------------------------------------------------------------
    def current_occupied_rule(m, j, t):
        return m.I_charge[j, t] <= m.I_max[j] * sum(m.x[i, j, t] for i in m.I)

    m.current_occupied = Constraint(m.J, m.T, rule=current_occupied_rule)

    # ------------------------------------------------------------------
    # C8 — Grid power cap
    # ------------------------------------------------------------------
    def grid_cap_rule(m, t):
        return sum(m.I_charge[j, t] * m.V[j] for j in m.J) <= m.P_max

    m.grid_cap = Constraint(m.T, rule=grid_cap_rule)

    # ------------------------------------------------------------------
    # C9 — SoC bounds
    # ------------------------------------------------------------------
    def soc_lb_rule(m, i, t):
        if t < value(m.arr[i]):
            return Constraint.Skip
        return m.s[i, t] >= m.s_init[i]

    def soc_ub_rule(m, i, t):
        return m.s[i, t] <= m.s_cap[i]

    m.soc_lb = Constraint(m.I, m.T, rule=soc_lb_rule)
    m.soc_ub = Constraint(m.I, m.T, rule=soc_ub_rule)


# ---------------------------------------------------------------------------
# Rolling-horizon constraints (single MPC step)
# ---------------------------------------------------------------------------

def add_rolling_constraints(
    m,
    data:        dict,
    assignments: dict,   # {car_i: port_j}
    soc_now:     dict,   # {car_i: float}
    t_start:     int,
) -> None:
    cars = sorted(assignments.keys())

    # ------------------------------------------------------------------
    # C1 — SoC dynamics  (soc_now is the live state entering t_start)
    # ------------------------------------------------------------------
    def soc_dynamics_rule(m, i, t):
        energy_in = m.phi[i, t] * data["V"][assignments[i]] * (data["delta_t"] / 1000.0)
        if t == t_start:
            return m.s[i, t] == soc_now[i] + energy_in
        return m.s[i, t] == m.s[i, t - 1] + energy_in

    m.soc_dynamics = Constraint(m.I, m.WIN, rule=soc_dynamics_rule)

    # ------------------------------------------------------------------
    # C2 — McCormick linearisation: phi[i,t] = x[i,t] * I_charge[assign[i],t]
    # ------------------------------------------------------------------
    def phi_ub_x_rule(m, i, t):
        return m.phi[i, t] <= data["I_max"][assignments[i]] * m.x[i, t]

    def phi_ub_I_rule(m, i, t):
        return m.phi[i, t] <= m.I_charge[assignments[i], t]

    def phi_lb_rule(m, i, t):
        return m.phi[i, t] >= m.I_charge[assignments[i], t] - data["I_max"][assignments[i]] * (1 - m.x[i, t])

    m.phi_ub_x = Constraint(m.I, m.WIN, rule=phi_ub_x_rule)
    m.phi_ub_I = Constraint(m.I, m.WIN, rule=phi_ub_I_rule)
    m.phi_lb   = Constraint(m.I, m.WIN, rule=phi_lb_rule)

    # ------------------------------------------------------------------
    # C3 — Departure condition
    # ------------------------------------------------------------------
    def depart_lb_rule(m, i, t):
        return m.s[i, t] >= m.s_target[i] * m.delta[i, t]

    def depart_ub_rule(m, i, t):
        return m.s[i, t] <= m.s_target[i] - m.epsilon + m.M_big * m.delta[i, t]

    def depart_monotone_rule(m, i, t):
        if t == t_start:
            return Constraint.Skip
        return m.delta[i, t] >= m.delta[i, t - 1]

    m.depart_lb       = Constraint(m.I, m.WIN, rule=depart_lb_rule)
    m.depart_ub       = Constraint(m.I, m.WIN, rule=depart_ub_rule)
    m.depart_monotone = Constraint(m.I, m.WIN, rule=depart_monotone_rule)

    # ------------------------------------------------------------------
    # C4 — Occupancy: all cars present at t_start, stay until departure
    # ------------------------------------------------------------------
    def occ_after_depart_rule(m, i, t):
        if t == t_start:
            return Constraint.Skip
        return m.x[i, t] <= 1 - m.delta[i, t - 1]

    def occ_presence_rule(m, i, t):
        departed_prev = 0 if t == t_start else m.delta[i, t - 1]
        return m.x[i, t] >= 1 - departed_prev

    m.occ_after_depart = Constraint(m.I, m.WIN, rule=occ_after_depart_rule)
    m.occ_presence     = Constraint(m.I, m.WIN, rule=occ_presence_rule)

    # ------------------------------------------------------------------
    # C5 — Port capacity: at most one car per port per time step
    # ------------------------------------------------------------------
    def port_capacity_rule(m, j, t):
        cars_on_j = [i for i in cars if assignments[i] == j]
        if not cars_on_j:
            return Constraint.Skip
        return sum(m.x[i, t] for i in cars_on_j) <= 1

    m.port_capacity = Constraint(m.J, m.WIN, rule=port_capacity_rule)

    # ------------------------------------------------------------------
    # C6 — Current only flows to occupied ports
    # ------------------------------------------------------------------
    def current_occupied_rule(m, j, t):
        cars_on_j = [i for i in cars if assignments[i] == j]
        if not cars_on_j:
            return m.I_charge[j, t] == 0
        return m.I_charge[j, t] <= data["I_max"][j] * sum(m.x[i, t] for i in cars_on_j)

    m.current_occupied = Constraint(m.J, m.WIN, rule=current_occupied_rule)

    # ------------------------------------------------------------------
    # C7 — Grid power cap
    # ------------------------------------------------------------------
    def grid_cap_rule(m, t):
        return sum(m.I_charge[j, t] * data["V"][j] for j in m.J) <= data["P_max"]

    m.grid_cap = Constraint(m.WIN, rule=grid_cap_rule)

    # ------------------------------------------------------------------
    # C8 — SoC bounds (charging only: SoC cannot fall below entry value)
    # ------------------------------------------------------------------
    def soc_lb_rule(m, i, t):
        return m.s[i, t] >= soc_now[i]

    def soc_ub_rule(m, i, t):
        return m.s[i, t] <= m.s_cap[i]

    m.soc_lb = Constraint(m.I, m.WIN, rule=soc_lb_rule)
    m.soc_ub = Constraint(m.I, m.WIN, rule=soc_ub_rule)
