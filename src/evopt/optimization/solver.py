from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Objective,
    Constraint, minimize, SolverFactory, value
)
from evopt.optimization.model import build_simplified_ev_model
from evopt.experiments.run_tiny_cost_case import make_small_data

data = make_small_data()
model = build_simplified_ev_model(data)

solver = SolverFactory("highs")
result = solver.solve(model, tee=True)

print("Status:", result.solver.status)
print("Termination:", result.solver.termination_condition)
print("Objective value:", value(model.obj))

for t in model.T:
    print(f"\nTime {t}")
    for j in model.J:
        print(f"  Port {j}: I = {value(model.I_charge[j, t]):.4f} A")