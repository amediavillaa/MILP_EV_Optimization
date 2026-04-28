"""
ev_charging_fcfs_milp.py — FCFS MILP for EV Charging Park
===========================================================
Decision variable : I_charge[j, t]  (A)  — the only direct control
Binary state      : y[i, j]         — port assignment (FCFS-constrained)
                    x[i, j, t]      — occupancy
                    delta[i, t]     — cumulative departure indicator
Continuous state  : s[i, t]         — state of charge (kWh)
Auxiliary         : phi[i, j, t]    — McCormick linearisation of
                                       x[i,j,t] * I_charge[j,t]

Objective:
    max  sum_{i,j,t}  phi[i,j,t] * V[j] * (dt/1000) * p_sell[t]
       - sum_{j,t}    I_charge[j,t] * V[j] * (dt/1000) * p_buy[t]

Revenue is earned per kWh delivered at the sell price prevailing in
each time step — it is not known at arrival but determined as charging
decisions are made across the horizon.
"""

from pyomo.environ import (
    ConcreteModel, RangeSet, Set, Param, Var,
    NonNegativeReals, Binary,
    Objective, Constraint,
    maximize, SolverFactory, value,
)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_ev_fcfs_model(data: dict) -> ConcreteModel:
    """
    Build the MILP from a data dictionary.

    Expected keys
    -------------
    J         int            number of ports
    T         int            number of time steps
    I         int            number of cars
    delta_t   float          hours per time step
    P_max     float          grid power cap (W)
    M_big     float          big-M constant (>= max s_cap across cars)
    epsilon   float          SoC tolerance for departure trigger (kWh)
    V         {j: float}     port voltage (V)
    I_max     {j: float}     max current per port (A)
    p_buy     {t: float}     electricity buy price (€/kWh)
    p_sell    {t: float}     electricity sell price to customers (€/kWh)
    arr       {i: int}       arrival time step (1-indexed)
    t_max     {i: int}       latest departure time step (hard deadline)
    s_init    {i: float}     initial SoC on arrival (kWh)
    s_target  {i: float}     target SoC that triggers departure (kWh)
    s_cap     {i: float}     battery capacity (kWh)
    """
    m = ConcreteModel()

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    m.J = RangeSet(1, data["J"])
    m.T = RangeSet(1, data["T"])
    m.I = RangeSet(1, data["I"])

    # Ordered pairs (i, i') where i' arrives strictly after i.
    # Used to enforce FCFS: i' cannot take port j while i is still there.
    m.FCFS_pairs = Set(
        initialize=[
            (i, ip)
            for i  in range(1, data["I"] + 1)
            for ip in range(1, data["I"] + 1)
            if ip != i and data["arr"][ip] > data["arr"][i]
        ],
        dimen=2,
    )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    m.delta_t = Param(initialize=data["delta_t"])
    m.P_max   = Param(initialize=data["P_max"])
    m.M_big   = Param(initialize=data["M_big"])
    m.epsilon = Param(initialize=data["epsilon"])

    m.V        = Param(m.J, initialize=data["V"])
    m.I_max    = Param(m.J, initialize=data["I_max"])
    m.p_buy    = Param(m.T, initialize=data["p_buy"])
    m.p_sell   = Param(m.T, initialize=data["p_sell"])
    m.arr      = Param(m.I, initialize=data["arr"])
    m.t_max    = Param(m.I, initialize=data["t_max"])
    m.s_init   = Param(m.I, initialize=data["s_init"])
    m.s_target = Param(m.I, initialize=data["s_target"])
    m.s_cap    = Param(m.I, initialize=data["s_cap"])

    # ------------------------------------------------------------------
    # Decision variable — the only direct control
    # ------------------------------------------------------------------
    m.I_charge = Var(m.J, m.T, domain=NonNegativeReals)

    # ------------------------------------------------------------------
    # Binary state variables
    # ------------------------------------------------------------------
    m.y     = Var(m.I, m.J,       domain=Binary)  # port assignment
    m.x     = Var(m.I, m.J, m.T,  domain=Binary)  # occupancy
    m.delta = Var(m.I, m.T,       domain=Binary)  # cumulative departure flag

    # ------------------------------------------------------------------
    # Continuous auxiliary variables
    # ------------------------------------------------------------------
    m.s   = Var(m.I, m.T, domain=NonNegativeReals)  # SoC (kWh)
    m.phi = Var(m.I, m.J, m.T, domain=NonNegativeReals)  # linearisation auxiliary

    # ------------------------------------------------------------------
    # Objective: maximise net profit
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # C1 — SoC dynamics
    #   s[i,t] = s[i,t-1] + sum_j phi[i,j,t] * V[j] * dt/1000
    #   boundary: s[i, arr[i]] starts from s_init[i]
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
            # Before arrival: SoC pinned to initial value, no charging
            return m.s[i, t] == m.s_init[i]

    m.soc_dynamics = Constraint(m.I, m.T, rule=soc_dynamics_rule)

    # ------------------------------------------------------------------
    # C2 — McCormick linearisation of phi[i,j,t] = x[i,j,t] * I_charge[j,t]
    #   Exact because x is binary and I_charge in [0, I_max[j]].
    # ------------------------------------------------------------------
    def phi_ub_x_rule(m, i, j, t):
        # if x=0 forces phi=0
        return m.phi[i, j, t] <= m.I_max[j] * m.x[i, j, t]

    def phi_ub_I_rule(m, i, j, t):
        # if x=1, phi cannot exceed actual current
        return m.phi[i, j, t] <= m.I_charge[j, t]

    def phi_lb_rule(m, i, j, t):
        # if x=1, forces phi >= I_charge (together with ub: phi = I_charge)
        return m.phi[i, j, t] >= m.I_charge[j, t] - m.I_max[j] * (1 - m.x[i, j, t])

    m.phi_ub_x = Constraint(m.I, m.J, m.T, rule=phi_ub_x_rule)
    m.phi_ub_I = Constraint(m.I, m.J, m.T, rule=phi_ub_I_rule)
    m.phi_lb   = Constraint(m.I, m.J, m.T, rule=phi_lb_rule)

    # ------------------------------------------------------------------
    # C3 — Departure condition
    #   delta[i,t]=1 iff s[i,t] >= s_target[i]; once set, stays set.
    # ------------------------------------------------------------------

    # 3a: delta=1 => SoC has reached target
    def depart_lb_rule(m, i, t):
        return m.s[i, t] >= m.s_target[i] * m.delta[i, t]

    # 3b: if delta=0, SoC has not yet reached target (prevents premature departure).
    #     s[i,t] <= s_target[i] - epsilon + M_big * delta[i,t]
    def depart_ub_rule(m, i, t):
        return m.s[i, t] <= m.s_target[i] - m.epsilon + m.M_big * m.delta[i, t]

    # 3c: delta is non-decreasing — departure is irreversible
    def depart_monotone_rule(m, i, t):
        if t == 1:
            return Constraint.Skip
        return m.delta[i, t] >= m.delta[i, t - 1]

    # 3d: delta=0 before car has arrived
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

    # 4a: can only occupy the assigned port
    def occ_assignment_rule(m, i, j, t):
        return m.x[i, j, t] <= m.y[i, j]

    # 4b: cannot be present after departure
    def occ_after_depart_rule(m, i, j, t):
        if t == 1:
            return Constraint.Skip  # delta[i,0]=0 by C3e, no constraint needed
        return m.x[i, j, t] <= 1 - m.delta[i, t - 1]

    # 4c: cannot be present before arrival
    def occ_before_arrival_rule(m, i, j, t):
        if t < value(m.arr[i]):
            return m.x[i, j, t] == 0
        return Constraint.Skip

    # 4d: if assigned and not yet departed, must be at assigned port.
    #     Linear lower bound: present >= assigned - departed_prev.
    #     Upper bounds are covered by occ_assignment (x <= y) and
    #     occ_after_depart (x <= 1 - delta[t-1]).
    def occ_presence_rule(m, i, t):
        if t < value(m.arr[i]):
            return Constraint.Skip  # occ_before_arr already forces x=0 here
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

    # 6a: each car gets at most one port (inequality allows rejection)
    def one_port_rule(m, i):
        return sum(m.y[i, j] for j in m.J) <= 1

    # 6b: car i' (later arrival) can only take port j if car i has already left.
    #     y[i',j] <= delta[i, arr[i']-1] + (1 - y[i,j])
    def fcfs_rule(m, i, ip, j):
        t_before_ip = value(m.arr[ip]) - 1
        if t_before_ip < 1:
            return Constraint.Skip  # i' arrives at t=1: no prior step, no conflict
        return m.y[ip, j] <= m.delta[i, t_before_ip] + (1 - m.y[i, j])

    m.one_port = Constraint(m.I,              rule=one_port_rule)
    m.fcfs     = Constraint(m.FCFS_pairs, m.J, rule=fcfs_rule)

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

    return m