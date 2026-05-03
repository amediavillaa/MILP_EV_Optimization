from pyomo.environ import ConcreteModel, SolverFactory, value
# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve(m: ConcreteModel, solver: str = "gurobi", verbose: bool = False):
    """Solve in-place. Swap solver='gurobi' for production use."""
    opt = SolverFactory(solver)
    result = opt.solve(m, tee=verbose)
    return result


# ---------------------------------------------------------------------------
# Solution printer
# ---------------------------------------------------------------------------

def print_solution(m: ConcreteModel) -> None:
    W = 70
    print("\n" + "=" * W)
    print("  EV Charging Park — FCFS MILP Solution")
    print("=" * W)

    revenue = sum(
        value(m.phi[i, j, t]) * value(m.V[j])
        * (value(m.delta_t) / 1000.0) * value(m.p_sell[t])
        for i in m.I for j in m.J for t in m.T
    )
    cost = sum(
        value(m.I_charge[j, t]) * value(m.V[j])
        * (value(m.delta_t) / 1000.0) * value(m.p_buy[t])
        for j in m.J for t in m.T
    )
    print(f"  Revenue     : €{revenue:.2f}")
    print(f"  Energy cost : €{cost:.2f}")
    print(f"  Net profit  : €{revenue - cost:.2f}")

    # --- per-car summary ---
    print()
    print(f"  {'Car':>3}  {'Port':>4}  {'Arr':>3}  {'Deadline':>8}  "
          f"{'SoC_0':>5}  {'SoC*':>5}  {'Status'}")
    print("  " + "-" * (W - 2))
    for i in m.I:
        port   = next((j for j in m.J if value(m.y[i, j]) > 0.5), None)
        served = value(m.delta[i, value(m.t_max[i])]) > 0.5
        if port is None:
            status = "REJECTED (no port available)"
        elif served:
            status = f"SERVED   (port {port})"
        else:
            status = f"PARTIAL  (port {port}, target not reached)"
        print(
            f"  {i:>3}  {str(port or '-'):>4}  {value(m.arr[i]):>3}  "
            f"{value(m.t_max[i]):>8}  "
            f"{value(m.s_init[i]):>5.1f}  {value(m.s_target[i]):>5.1f}  "
            f"{status}"
        )

    # --- charging schedule ---
    print(f"\n  Charging schedule (kW per port per time step):")
    print(f"  {'Port':<5}", end="")
    for t in m.T:
        print(f"   t{t:<3}", end="")
    print()
    for j in m.J:
        print(f"  {j:<5}", end="")
        for t in m.T:
            kw = value(m.I_charge[j, t]) * value(m.V[j]) / 1000.0
            print(f"  {kw:>5.2f}", end="")
        print()

    # --- SoC trajectory ---
    print(f"\n  SoC trajectory per car (kWh)  [* = target reached]:")
    print(f"  {'Car':<4}", end="")
    for t in m.T:
        print(f"   t{t:<4}", end="")
    print()
    for i in m.I:
        print(f"  {i:<4}", end="")
        for t in m.T:
            s    = value(m.s[i, t])
            flag = "*" if s >= value(m.s_target[i]) - value(m.epsilon) else " "
            print(f"  {s:>5.2f}{flag}", end="")
        print()

    print("=" * W)


# ---------------------------------------------------------------------------
# Rolling-horizon solution printer
# ---------------------------------------------------------------------------

def print_rolling_solution(data: dict, res, horizon) -> None:
    """Print full metrics for one simulation result (rolling MPC or rule-based)."""
    W = 70
    T = data["T"]

    label = f"H={horizon}" if isinstance(horizon, int) else str(horizon)
    print("\n" + "=" * W)
    print(f"  EV Charging Park — {label}")
    print("=" * W)

    print(f"  Revenue     : €{res.revenue:.2f}")
    print(f"  Energy cost : €{res.cost:.2f}")
    print(f"  Net profit  : €{res.net_profit:.2f}")

    # --- per-car summary ---
    print()
    print(f"  {'Car':>3}  {'Port':>4}  {'Arr':>3}  {'Deadline':>8}  "
          f"{'SoC_0':>5}  {'SoC*':>5}  {'Status'}")
    print("  " + "-" * (W - 2))
    for i in range(1, data["I"] + 1):
        port = res.assignments.get(i)
        if i in res.rejected:
            status = "REJECTED (no port available)"
        elif i in res.departed:
            status = f"SERVED   (port {port})"
        else:
            status = f"PARTIAL  (port {port}, target not reached)"
        print(
            f"  {i:>3}  {str(port or '-'):>4}  {data['arr'][i]:>3}  "
            f"{data['t_max'][i]:>8}  "
            f"{data['s_init'][i]:>5.1f}  {data['s_target'][i]:>5.1f}  "
            f"{status}"
        )

    # --- charging schedule ---
    print(f"\n  Charging schedule (kW per port per time step):")
    print(f"  {'Port':<5}", end="")
    for t in range(1, T + 1):
        print(f"   t{t:<3}", end="")
    print()
    for j in range(1, data["J"] + 1):
        print(f"  {j:<5}", end="")
        for t in range(1, T + 1):
            amps = res.current_applied.get((j, t), 0.0)
            kw   = amps * data["V"][j] / 1000.0
            print(f"  {kw:>5.2f}", end="")
        print()

    # --- SoC trajectory ---
    print(f"\n  SoC trajectory per car (kWh)  [* = target reached]:")
    print(f"  {'Car':<4}", end="")
    for t in range(1, T + 1):
        print(f"   t{t:<4}", end="")
    print()
    for i in range(1, data["I"] + 1):
        print(f"  {i:<4}", end="")
        for t in range(1, T + 1):
            s    = res.soc_trajectory[i][t]   # index 0 = initial, index t = after step t
            flag = "*" if s >= data["s_target"][i] - data["epsilon"] else " "
            print(f"  {s:>5.2f}{flag}", end="")
        print()

    print("=" * W)


# ---------------------------------------------------------------------------
# Horizon comparison
# ---------------------------------------------------------------------------

def run_horizon_comparison(
    data:     dict  = None,
    solver:   str   = "gurobi",
    horizons: list  = None,
) -> None:
    """
    Solve the offline MILP and run rolling-horizon MPC for each H.
    Prints full metrics for every model, then a summary comparison table.

    Parameters
    ----------
    data     scenario dict; defaults to SAMPLE_DATA when None
    solver   Pyomo solver name
    horizons list of H values to sweep; defaults to [1, 2, 4, T]
    """
    from evopt.experiments.run_tiny_cost_case import SAMPLE_DATA
    from evopt.experiments.runner import run_rolling_horizon, run_equal_allocation
    from evopt.optimization.model import build_ev_fcfs_model

    if data is None:
        data = SAMPLE_DATA

    T  = data["T"]
    Hs = horizons if horizons is not None else [1, 2, 4, T]

    # --- offline optimal ---
    m_off = build_ev_fcfs_model(data)
    solve(m_off, solver=solver)
    print_solution(m_off)

    opt_revenue = sum(
        value(m_off.phi[i, j, t]) * value(m_off.V[j])
        * (value(m_off.delta_t) / 1000.0) * value(m_off.p_sell[t])
        for i in m_off.I for j in m_off.J for t in m_off.T
    )
    opt_cost = sum(
        value(m_off.I_charge[j, t]) * value(m_off.V[j])
        * (value(m_off.delta_t) / 1000.0) * value(m_off.p_buy[t])
        for j in m_off.J for t in m_off.T
    )
    opt_profit = opt_revenue - opt_cost

    # --- equal allocation ---
    res_eq = run_equal_allocation(data)
    print_rolling_solution(data, res_eq, horizon="equal alloc")

    # --- rolling horizons ---
    results = {}
    for H in Hs:
        res = run_rolling_horizon(data, horizon=H, solver_name=solver)
        results[H] = res
        print_rolling_solution(data, res, H)

    # --- summary table ---
    W = 72
    print("\n" + "=" * W)
    print("  Summary — All Models")
    print("=" * W)
    print(f"\n  {'Model':>14}  {'Revenue':>9}  {'Cost':>9}  {'Net profit':>11}  "
          f"{'Gap':>8}  {'Served':>7}  {'Rejected':>9}")
    print("  " + "-" * (W - 2))
    print(f"  {'Offline':>14}  €{opt_revenue:>7.2f}  €{opt_cost:>7.2f}  "
          f"€{opt_profit:>9.2f}  {'—':>8}  {'—':>7}  {'—':>9}")

    eq_gap = opt_profit - res_eq.net_profit
    print(f"  {'Equal alloc':>14}  €{res_eq.revenue:>7.2f}  €{res_eq.cost:>7.2f}  "
          f"€{res_eq.net_profit:>9.2f}  "
          f"{'Δ€'+f'{eq_gap:.2f}':>8}  "
          f"{len(res_eq.departed):>7}  {len(res_eq.rejected):>9}")

    for H, res in results.items():
        label = f"H={H}" if H < T else f"H=T={T}"
        gap   = opt_profit - res.net_profit
        print(f"  {label:>14}  €{res.revenue:>7.2f}  €{res.cost:>7.2f}  "
              f"€{res.net_profit:>9.2f}  "
              f"{'Δ€'+f'{gap:.2f}':>8}  "
              f"{len(res.departed):>7}  {len(res.rejected):>9}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_horizon_comparison()