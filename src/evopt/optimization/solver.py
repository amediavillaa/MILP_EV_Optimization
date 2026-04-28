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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from evopt.experiments.run_tiny_cost_case import SAMPLE_DATA
    from evopt.optimization.model import build_ev_fcfs_model

    print("Building model...")
    m = build_ev_fcfs_model(SAMPLE_DATA)

    print("Solving...")
    solve(m, solver="gurobi", verbose=True)

    print_solution(m)