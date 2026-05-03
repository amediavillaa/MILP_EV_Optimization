"""
runner.py — Rolling-horizon simulation loop for EV charging park
=================================================================
At each time step t:
  1. Arrivals  — any car with arr[i] == t is assigned to the lowest-index
                 free port (greedy).  If no port is free the car is rejected.
  2. Solve     — build and solve a rolling MILP over [t, t + H - 1] for all
                 currently present cars.
  3. Apply     — read s[i, t] from the solved model to update live SoC.
  4. Depart    — any car with delta[i, t] == 1 is removed and its port freed.
"""

from dataclasses import dataclass, field

from pyomo.environ import value

from evopt.optimization.model import build_rolling_model, compute_equal_allocation
from evopt.optimization.solver import solve


@dataclass
class SimResults:
    soc_final:       dict    # {car_i: float}       final SoC at end of horizon
    soc_trajectory:  dict    # {car_i: [float]}     SoC recorded after each step (index 0 = initial)
    assignments:     dict    # {car_i: port_j}      port assigned on arrival
    departed:        set     # cars that reached s_target
    rejected:        set     # cars rejected because all ports were busy
    net_profit:      float   # total revenue − cost
    revenue:         float   # total revenue
    cost:            float   # total energy cost
    current_applied: dict    # {(port_j, step_t): float}  applied charging current (A)


def run_rolling_horizon(
    data:        dict,
    horizon:     int,
    solver_name: str  = "gurobi",
    verbose:     bool = False,
) -> SimResults:
    """
    Simulate the charging park under rolling-horizon MPC.

    Parameters
    ----------
    data        scenario dict (same schema as build_ev_fcfs_model)
    horizon     MPC lookahead length H (steps)
    solver_name Pyomo solver name
    verbose     print per-step events when True
    """
    T = data["T"]
    J = data["J"]

    soc             = {i: data["s_init"][i] for i in range(1, data["I"] + 1)}
    assignments     = {}   # {car_i: port_j}
    departed        = set()
    rejected        = set()
    revenue_total   = 0.0
    cost_total      = 0.0
    current_applied = {}   # {(j, t): float}

    soc_traj = {i: [data["s_init"][i]] for i in range(1, data["I"] + 1)}

    for t in range(1, T + 1):

        # ------------------------------------------------------------------
        # 1. Process arrivals
        # ------------------------------------------------------------------
        for i in range(1, data["I"] + 1):
            if data["arr"][i] == t and i not in assignments and i not in rejected:
                occupied = {assignments[k] for k in assignments if k not in departed}
                free     = [j for j in range(1, J + 1) if j not in occupied]
                if free:
                    assignments[i] = free[0]
                    if verbose:
                        print(f"  t={t}: car {i} → port {free[0]}")
                else:
                    rejected.add(i)
                    if verbose:
                        print(f"  t={t}: car {i} REJECTED (all ports busy)")

        # ------------------------------------------------------------------
        # 2. Build present-car map
        # ------------------------------------------------------------------
        present     = {i: assignments[i] for i in assignments if i not in departed}
        present_soc = {i: soc[i] for i in present}

        if not present:
            for i in range(1, data["I"] + 1):
                soc_traj[i].append(soc[i])
            continue

        # ------------------------------------------------------------------
        # 3. Solve rolling model for window [t, t + H - 1]
        # ------------------------------------------------------------------
        m = build_rolling_model(data, t, horizon, present, present_soc)
        result = solve(m, solver=solver_name, verbose=False)

        # ------------------------------------------------------------------
        # 4. Apply first-step decisions: advance SoC from model output
        # ------------------------------------------------------------------
        for i in present:
            soc[i] = value(m.s[i, t])

        dt = data["delta_t"]
        step_revenue = sum(
            value(m.phi[i, t]) * data["V"][assignments[i]] * (dt / 1000.0) * data["p_sell"][t]
            for i in present
        )
        step_cost = sum(
            value(m.I_charge[j, t]) * data["V"][j] * (dt / 1000.0) * data["p_buy"][t]
            for j in range(1, J + 1)
        )
        revenue_total += step_revenue
        cost_total    += step_cost

        for j in range(1, J + 1):
            current_applied[(j, t)] = value(m.I_charge[j, t])

        # ------------------------------------------------------------------
        # 5. Process departures
        # ------------------------------------------------------------------
        for i in list(present.keys()):
            if value(m.delta[i, t]) > 0.5:
                departed.add(i)
                if verbose:
                    print(f"  t={t}: car {i} departed  "
                          f"SoC {soc[i]:.2f} / {data['s_target'][i]:.2f} kWh")

        for i in range(1, data["I"] + 1):
            soc_traj[i].append(soc[i])

    return SimResults(
        soc_final       = soc,
        soc_trajectory  = soc_traj,
        assignments     = assignments,
        departed        = departed,
        rejected        = rejected,
        net_profit      = revenue_total - cost_total,
        revenue         = revenue_total,
        cost            = cost_total,
        current_applied = current_applied,
    )


def run_equal_allocation(
    data:    dict,
    verbose: bool = False,
) -> SimResults:
    """
    Simulate the charging park using the equal-allocation rule.

    At each step the available grid capacity is split equally among present
    cars (capped by port limit and energy needed to reach s_target).
    Port assignment on arrival is greedy (lowest-index free port), identical
    to the rolling-horizon runner.

    Parameters
    ----------
    data    scenario dict (same schema as build_ev_fcfs_model)
    verbose print per-step events when True
    """
    T = data["T"]
    J = data["J"]

    soc             = {i: data["s_init"][i] for i in range(1, data["I"] + 1)}
    assignments     = {}
    departed        = set()
    rejected        = set()
    revenue_total   = 0.0
    cost_total      = 0.0
    current_applied = {}

    soc_traj = {i: [data["s_init"][i]] for i in range(1, data["I"] + 1)}

    for t in range(1, T + 1):

        # ------------------------------------------------------------------
        # 1. Process arrivals (identical greedy logic to rolling runner)
        # ------------------------------------------------------------------
        for i in range(1, data["I"] + 1):
            if data["arr"][i] == t and i not in assignments and i not in rejected:
                occupied = {assignments[k] for k in assignments if k not in departed}
                free     = [j for j in range(1, J + 1) if j not in occupied]
                if free:
                    assignments[i] = free[0]
                    if verbose:
                        print(f"  t={t}: car {i} → port {free[0]}")
                else:
                    rejected.add(i)
                    if verbose:
                        print(f"  t={t}: car {i} REJECTED (all ports busy)")

        # ------------------------------------------------------------------
        # 2. Apply equal-allocation rule
        # ------------------------------------------------------------------
        present     = {i: assignments[i] for i in assignments if i not in departed}
        present_soc = {i: soc[i] for i in present}

        currents, soc_next = compute_equal_allocation(data, present, present_soc)

        for i in present:
            soc[i] = soc_next[i]

        for j, amps in currents.items():
            current_applied[(j, t)] = amps

        # ------------------------------------------------------------------
        # 3. Accumulate financials
        # ------------------------------------------------------------------
        dt = data["delta_t"]
        for i in present:
            j    = assignments[i]
            amps = currents[j]
            energy = amps * data["V"][j] * (dt / 1000.0)
            revenue_total += energy * data["p_sell"][t]
            cost_total    += energy * data["p_buy"][t]

        # ------------------------------------------------------------------
        # 4. Process departures
        # ------------------------------------------------------------------
        for i in list(present.keys()):
            if soc[i] >= data["s_target"][i] - data["epsilon"]:
                departed.add(i)
                if verbose:
                    print(f"  t={t}: car {i} departed  "
                          f"SoC {soc[i]:.2f} / {data['s_target'][i]:.2f} kWh")

        for i in range(1, data["I"] + 1):
            soc_traj[i].append(soc[i])

    return SimResults(
        soc_final       = soc,
        soc_trajectory  = soc_traj,
        assignments     = assignments,
        departed        = departed,
        rejected        = rejected,
        net_profit      = revenue_total - cost_total,
        revenue         = revenue_total,
        cost            = cost_total,
        current_applied = current_applied,
    )
