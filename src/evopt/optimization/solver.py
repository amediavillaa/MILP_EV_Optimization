from pyomo.environ import ConcreteModel, SolverFactory
from pyomo.opt import TerminationCondition

# Hard cap per solve. Normal solves complete in < 60 ms (worst observed: H=12, 12 ports).
# This prevents degenerate simplex instances from stalling a full episode.
_HIGHS_TIME_LIMIT_S = 2.0


def solve(m: ConcreteModel, solver: str = "gurobi", verbose: bool = False):
    """Solve in-place. Swap solver='gurobi' for production use."""
    opt = SolverFactory(solver)
    options = {"time_limit": _HIGHS_TIME_LIMIT_S} if solver == "highs" else {}
    result = opt.solve(m, tee=verbose, options=options)
    tc = result.solver.termination_condition
    if tc in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
        raise RuntimeError(f"LP infeasible: {tc}")
    return result
