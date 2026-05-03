"""
model.py — MILP model builders for EV charging park
=====================================================
Each builder creates a ConcreteModel, declares sets and parameters,
then delegates variables, objective, and constraints to the dedicated
modules so this file stays focused on structure and data wiring.

build_ev_fcfs_model(data)
    Full offline MILP over [1, T]. Port assignments and charging
    schedule are optimised jointly. Use for the clairvoyant benchmark.

build_rolling_model(data, t_start, horizon, assignments, soc_now)
    Single MPC step over [t_start, t_start + H - 1]. Port assignments
    are fixed externally (greedy on arrival); only charging rates are
    optimised. Called once per simulation step by the rolling runner.

compute_equal_allocation(data, assignments, soc_now)
    Rule-based equal-share controller. No optimisation — divides P_max
    equally among present cars and returns the resulting currents and
    next-step SoC directly.
"""

from pyomo.environ import ConcreteModel, RangeSet, Set, Param

from evopt.optimization.variables   import add_offline_variables,   add_rolling_variables
from evopt.optimization.objective   import add_offline_objective,   add_rolling_objective
from evopt.optimization.constraints import add_offline_constraints, add_rolling_constraints


def _add_physical_params(m, data: dict) -> None:
    """Add the physical parameters shared by both model builders."""
    m.delta_t = Param(initialize=data["delta_t"])
    m.P_max   = Param(initialize=data["P_max"])
    m.M_big   = Param(initialize=data["M_big"])
    m.epsilon = Param(initialize=data["epsilon"])
    m.V       = Param(m.J, initialize=data["V"])
    m.I_max   = Param(m.J, initialize=data["I_max"])


# ---------------------------------------------------------------------------
# Full offline MILP
# ---------------------------------------------------------------------------

def build_ev_fcfs_model(data: dict) -> ConcreteModel:
    """
    Build the offline MILP from a scenario data dictionary.

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
    t_max     {i: int}       latest departure time step
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
    # Used by the FCFS constraint to prevent i' taking port j while i is there.
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
    _add_physical_params(m, data)

    m.p_buy    = Param(m.T, initialize=data["p_buy"])
    m.p_sell   = Param(m.T, initialize=data["p_sell"])
    m.arr      = Param(m.I, initialize=data["arr"])
    m.t_max    = Param(m.I, initialize=data["t_max"])
    m.s_init   = Param(m.I, initialize=data["s_init"])
    m.s_target = Param(m.I, initialize=data["s_target"])
    m.s_cap    = Param(m.I, initialize=data["s_cap"])

    # ------------------------------------------------------------------
    # Variables, objective, constraints
    # ------------------------------------------------------------------
    add_offline_variables(m)
    add_offline_objective(m)
    add_offline_constraints(m)

    return m


# ---------------------------------------------------------------------------
# Rolling-horizon model builder (single MPC step)
# ---------------------------------------------------------------------------

def build_rolling_model(
    data:        dict,
    t_start:     int,
    horizon:     int,
    assignments: dict,   # {car_i: port_j}  — fixed, set on arrival
    soc_now:     dict,   # {car_i: float}   — live SoC at the start of t_start
) -> ConcreteModel | None:
    """
    Build one MPC step model for window [t_start, t_start + horizon - 1].

    Returns None when no cars are present.
    """
    t_end = min(t_start + horizon - 1, data["T"])
    cars  = sorted(assignments.keys())

    if not cars:
        return None

    m = ConcreteModel()

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    m.WIN = RangeSet(t_start, t_end)
    m.I   = Set(initialize=cars)
    m.J   = RangeSet(1, data["J"])

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    _add_physical_params(m, data)

    m.p_buy    = Param(m.WIN, initialize={t: data["p_buy"][t]  for t in range(t_start, t_end + 1)})
    m.p_sell   = Param(m.WIN, initialize={t: data["p_sell"][t] for t in range(t_start, t_end + 1)})
    m.s_target = Param(m.I,   initialize={i: data["s_target"][i] for i in cars})
    m.s_cap    = Param(m.I,   initialize={i: data["s_cap"][i]    for i in cars})

    # ------------------------------------------------------------------
    # Variables, objective, constraints
    # ------------------------------------------------------------------
    add_rolling_variables(m)
    add_rolling_objective(m, data, assignments)
    add_rolling_constraints(m, data, assignments, soc_now, t_start)

    return m


# ---------------------------------------------------------------------------
# Rule-based model: equal allocation
# ---------------------------------------------------------------------------

def compute_equal_allocation(
    data:        dict,
    assignments: dict,   # {car_i: port_j}  — currently present, non-departed cars
    soc_now:     dict,   # {car_i: float}   — live SoC at the start of this step
) -> tuple[dict, dict]:
    """
    Compute one time step of equal-share charging.

    Divides P_max equally across all present cars. Each car's share is
    further capped by its port's current limit and the energy still needed
    to reach s_target. Unused capacity is NOT redistributed.

    Returns
    -------
    currents  {port_j: amps}   charging current to apply at each port
    soc_next  {car_i: float}   SoC after this time step
    """
    cars = sorted(assignments.keys())
    N    = len(cars)

    currents = {j: 0.0 for j in range(1, data["J"] + 1)}
    soc_next = dict(soc_now)

    if N == 0:
        return currents, soc_next

    dt      = data["delta_t"]
    P_share = data["P_max"] / N

    for i in cars:
        j = assignments[i]
        V = data["V"][j]

        P_port   = V * data["I_max"][j]
        P_needed = max(0.0, (data["s_target"][i] - soc_now[i]) / dt * 1000.0)

        P_i = min(P_share, P_port, P_needed)
        I_i = P_i / V

        currents[j] = I_i
        soc_next[i] = soc_now[i] + I_i * V * (dt / 1000.0)

    return currents, soc_next
