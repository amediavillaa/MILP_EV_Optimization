"""
model.py — LP model builders for EV charging park with BESS
=============================================================
Implements the LP formulation where port assignments and occupancy
are precomputed externally, and the sole decision variable is the
charging current I_{j,t} at every port (including the BESS at j=J+1).

build_ev_lp_model(data)
    Full offline LP over [1, T].  Minimises charging cost + battery
    cost.  Assignments, occupancy, and SoC-dependent rate curves are
    all supplied as parameters in *data*.

build_rolling_model(data, t_start, horizon, assignments, soc_now,
                    socb_now)
    Single MPC step over [t_start, t_start + H - 1].  Same cost
    structure; car SoC and BESS SoC are warm-started from the caller.
"""

from pyomo.environ import (
    ConcreteModel,
    NonNegativeReals,
    Param,
    RangeSet,
    Set,
    Var,
)

from milp_ev_opt.optimization.objective   import (
    add_offline_profit_objective,
    add_rolling_profit_objective,
)
from milp_ev_opt.optimization.constraints import (
    add_offline_constraints,
    add_rolling_constraints,
    add_must_serve_constraints,
)


# ======================================================================
#  Helpers
# ======================================================================

def _build_occupancy(data: dict) -> dict:
    """
    Derive z[j,t] from assignments, arrival, and departure times.
    z[j,t] = 1  iff some car i with y_{i,j}=1 satisfies arr[i] <= t <= dep[i].
    """
    z = {}
    for j in range(1, data["J"] + 1):
        for t in range(1, data["T"] + 1):
            z[j, t] = 0
    for i, j in data["assignments"].items():
        for t in range(data["arr"][i], data["dep"][i] + 1):
            z[j, t] = 1
    return z


# ======================================================================
#  Full offline LP
# ======================================================================

def build_ev_lp_model(data: dict, eta_bess: float = 0.95) -> ConcreteModel:
    """
    Build the offline LP from a scenario data dictionary.

    Expected keys
    -------------
    J           int              number of EV charging ports (BESS = J+1)
    T           int              number of time steps
    I           int              number of cars
    delta_t     float            time-step duration (h)
    P_max       float            grid power cap (W)

    V           {j: float}       port voltage (V), j = 1 … J+1
    I_max       {j: float}       max current per EV port (A), j = 1 … J

    I_high      float            BESS max charging current (A, ≥ 0)
    I_low       float            BESS max discharging current (A, ≥ 0)

    p_buy       {t: float}       electricity buy price  (€/kWh)
    p_sell      {t: float}       electricity sell price  (€/kWh)

    L           {t: float}       background load at step t (W)

    assignments {i: j}           car → port mapping (precomputed)
    arr         {i: int}         arrival time step of car i
    dep         {i: int}         departure time step of car i

    s_init      {i: float}       initial car SoC on arrival (kWh)
    s_cap       {i: float}       car battery capacity / SoC_max (kWh)
    s_min       {i: float}       car minimum SoC (kWh)
    P_car_max   {i: float}       max charging power per car (W)
    r_car       {(i,t): float}   SoC-dependent charging ratio ∈ [0,1]
                                 (set to 1.0 everywhere if ignored)

    SoCB_init   float            initial BESS SoC (kWh)
    SoCB_min    float            BESS minimum SoC (kWh)
    SoCB_max    float            BESS maximum SoC (kWh)
    r_bess_ch   {t: float}       BESS charging ratio ∈ [0,1]
    r_bess_dis  {t: float}       BESS discharging ratio ∈ [0,1]
    """
    m = ConcreteModel()

    J      = data["J"]
    j_bess = J + 1

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    m.T     = RangeSet(1, data["T"])
    m.J_ev  = RangeSet(1, J)
    m.J_all = RangeSet(1, j_bess)
    m.I     = RangeSet(1, data["I"])

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    m.delta_t   = Param(initialize=data["delta_t"])
    m.P_max     = Param(initialize=data["P_max"])
    m.V         = Param(m.J_all, initialize=data["V"])
    m.I_max     = Param(m.J_ev,  initialize=data["I_max"])
    m.I_high    = Param(initialize=data["I_high"])
    m.I_low     = Param(initialize=data["I_low"])
    m.p_buy       = Param(m.T, initialize=data["p_buy"])
    m.p_sell      = Param(m.T, initialize=data["p_sell"])
    m.L           = Param(m.T, initialize=data["L"])
    m.SoCB_min    = Param(initialize=data["SoCB_min"])
    m.SoCB_max    = Param(initialize=data["SoCB_max"])
    m.SoCB_init   = Param(initialize=data["SoCB_init"])

    z_data = data.get("z") or _build_occupancy(data)
    m.z = Param(m.J_ev, m.T, initialize=z_data, default=0)

    m.s_init    = Param(m.I, initialize=data["s_init"])
    m.s_cap     = Param(m.I, initialize=data["s_cap"])
    m.s_min     = Param(m.I, initialize=data["s_min"])
    m.P_car_max = Param(m.I, initialize=data["P_car_max"])

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    m.I_ev       = Var(m.J_ev, m.T, within=NonNegativeReals)
    m.I_bess_ch  = Var(m.T, within=NonNegativeReals)
    m.I_bess_dis = Var(m.T, within=NonNegativeReals)
    m.SoCB       = Var(m.T, within=NonNegativeReals)
    m.soc_car    = Var(m.I, m.T, within=NonNegativeReals)

    # ------------------------------------------------------------------
    # Objective and constraints
    # ------------------------------------------------------------------
    add_offline_profit_objective(m, j_bess, eta_bess=eta_bess)
    add_offline_constraints(m, data, j_bess, eta_bess=eta_bess)

    return m


# ======================================================================
#  Rolling-horizon model (single MPC step)
# ======================================================================

def build_rolling_model(
    data:        dict,
    t_start:     int,
    horizon:     int,
    assignments: dict,   # {car_i: port_j}  — currently docked cars
    soc_now:     dict,   # {car_i: float}   — car SoC at start of t_start
    socb_now:    float,  #                    BESS SoC at start of t_start
    bare:        bool  = False,  # skip must-serve constraints (infeasibility fallback)
    eta_bess:    float = 0.95,
) -> ConcreteModel | None:
    """
    Build one MPC step over [t_start, t_start + horizon - 1].

    Returns None when no cars are present.
    """
    J      = data["J"]
    j_bess = J + 1
    t_end  = min(t_start + horizon - 1, data["T"])
    cars   = sorted(assignments.keys())

    if not cars:
        return None

    m = ConcreteModel()

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    m.WIN   = RangeSet(t_start, t_end)
    m.I     = Set(initialize=cars)
    m.J_ev  = RangeSet(1, J)
    m.J_all = RangeSet(1, j_bess)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    m.delta_t   = Param(initialize=data["delta_t"])
    m.P_max     = Param(initialize=data["P_max"])
    m.V         = Param(m.J_all, initialize=data["V"])
    m.I_max     = Param(m.J_ev,  initialize=data["I_max"])
    m.I_high    = Param(initialize=data["I_high"])
    m.I_low     = Param(initialize=data["I_low"])
    m.SoCB_min  = Param(initialize=data["SoCB_min"])
    m.SoCB_max  = Param(initialize=data["SoCB_max"])
    m.p_buy       = Param(m.WIN, initialize={
        t: data["p_buy"][t] for t in range(t_start, t_end + 1)
    })
    m.p_sell      = Param(m.WIN, initialize={
        t: data["p_sell"][t] for t in range(t_start, t_end + 1)
    })
    m.L         = Param(m.WIN, initialize={
        t: data["L"][t] for t in range(t_start, t_end + 1)
    })
    m.s_cap     = Param(m.I, initialize={i: data["s_cap"][i] for i in cars})
    m.s_min     = Param(m.I, initialize={i: data["s_min"][i] for i in cars})
    m.P_car_max = Param(m.I, initialize={i: data["P_car_max"][i] for i in cars})

    z_win = {}
    for j in range(1, J + 1):
        for t in range(t_start, t_end + 1):
            z_win[j, t] = 0
    for i in cars:
        j     = assignments[i]
        dep_i = data["dep"][i]
        for t in range(t_start, min(dep_i, t_end) + 1):
            z_win[j, t] = 1
    m.z = Param(m.J_ev, m.WIN, initialize=z_win, default=0)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    m.I_ev       = Var(m.J_ev, m.WIN, within=NonNegativeReals)
    m.I_bess_ch  = Var(m.WIN, within=NonNegativeReals)
    m.I_bess_dis = Var(m.WIN, within=NonNegativeReals)
    m.SoCB       = Var(m.WIN, within=NonNegativeReals)
    m.soc_car    = Var(m.I, m.WIN, within=NonNegativeReals)

    # ------------------------------------------------------------------
    # Objective and constraints
    # ------------------------------------------------------------------
    add_rolling_profit_objective(m, j_bess, eta_bess=eta_bess)
    add_rolling_constraints(m, data, assignments, soc_now, socb_now, t_start, j_bess,
                            eta_bess=eta_bess)
    if not bare:
        add_must_serve_constraints(m, data, soc_now, t_start, t_end)

    return m

