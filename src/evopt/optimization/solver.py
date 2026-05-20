from pyomo.environ import ConcreteModel, SolverFactory


def solve(m: ConcreteModel, solver: str = "gurobi", verbose: bool = False):
    """Solve in-place. Swap solver='gurobi' for production use."""
    opt = SolverFactory(solver)
    result = opt.solve(m, tee=verbose)
    return result
