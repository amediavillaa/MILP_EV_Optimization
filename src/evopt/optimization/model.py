from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Objective,
    Constraint, minimize, SolverFactory, value
)

def build_simplified_ev_model(data):
    m = ConcreteModel()

    # Sets
    m.J = RangeSet(1, data["J"])
    m.T = RangeSet(1, data["T"])
    m.I = RangeSet(1, data["I"])

    # Parameters
    m.delta_t = Param(initialize=data["delta_t"])
    m.P_max = Param(initialize=data["P_max"])

    m.V = Param(m.J, initialize=data["V"])
    m.I_max = Param(m.J, initialize=data["I_max"])
    m.L = Param(m.T, initialize=data["L"])
    m.p_buy = Param(m.T, initialize=data["p_buy"])
    m.E = Param(m.I, initialize=data["E"])
    m.arrival = Param(m.I, initialize=data["arrival"])
    m.departure = Param(m.I, initialize=data["departure"])
    m.y = Param(m.I, m.J, initialize=data["y"])
    m.z = Param(m.J, m.T, initialize=data["z"])

    # Decision variables
    m.I_charge = Var(m.J, m.T, domain=NonNegativeReals)

    # Objective: charging cost only
    def obj_rule(m):
        return sum(
            (m.V[j] * m.I_charge[j, t] / 1000.0) * m.delta_t * m.p_buy[t]
            for j in m.J for t in m.T
        )
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # Constraint 1: port current limit with occupancy
    def port_limit_rule(m, j, t):
        return m.I_charge[j, t] <= m.z[j, t] * m.I_max[j]
    m.port_limit = Constraint(m.J, m.T, rule=port_limit_rule)

    # Constraint 2: site power limit
    def site_limit_rule(m, t):
        return m.L[t] + sum(m.V[j] * m.I_charge[j, t] / 1000.0 for j in m.J) <= m.P_max
    m.site_limit = Constraint(m.T, rule=site_limit_rule)

    # Constraint 3: energy delivery requirement
    def energy_rule(m, i):
        return sum(
            m.y[i, j] * (m.V[j] * m.I_charge[j, t] / 1000.0) * m.delta_t
            for j in m.J
            for t in m.T
            if value(m.arrival[i]) <= t <= value(m.departure[i])
        ) >= m.E[i]
    m.energy_req = Constraint(m.I, rule=energy_rule)

    return m