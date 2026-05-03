# Test Suite Design — EV Charging Optimization
_Date: 2026-05-03_

## Goal

Add tests covering correctness, physical/model invariants, and robustness edge cases for the rolling-horizon MPC and equal-allocation runners.

## Scope

- Solver: Gurobi only (for now)
- Numerical tolerance: 1% (`1e-2`) for floating-point assertions; exact (`1e-6`) for accounting identities
- Test location: `src/evopt/tests/`

---

## File Structure

```
src/evopt/tests/
├── conftest.py          # shared pytest fixtures (minimal scenarios)
├── test_correctness.py  # numerical result verification
├── test_invariants.py   # physical/model constraints always hold
└── test_edge_cases.py   # robustness under unusual inputs
```

---

## Shared Fixtures (`conftest.py`)

Three minimal `data` dicts matching the schema used by `runner.py` and `solver.py`. No new abstractions — plain dicts as pytest fixtures.

| Fixture | Setup | Purpose |
|---|---|---|
| `single_car` | 1 port, 1 car, 3 steps, uniform prices | Hand-calculable ground truth for correctness |
| `port_saturation` | 1 port, 3 cars all arriving at t=1 | Rejection and greedy assignment logic |
| `sequential_arrival` | 2 ports, 3 cars: car 1 departs mid-run, car 3 takes its port | Departure and port reuse |

---

## Correctness Tests (`test_correctness.py`)

Verify that known-good numerical outputs are reproduced within 1% tolerance. Scenarios: `single_car` and `port_saturation`.

| Test | Scenario | What's checked |
|---|---|---|
| `test_single_car_final_soc` | `single_car` | Final SoC matches hand-calculated value (1e-2 tolerance) |
| `test_profit_identity` | `single_car` | `net_profit == revenue - cost` to 1e-6 |
| `test_soc_trajectory_length` | `single_car` | `soc_trajectory[i]` has `T+1` entries |
| `test_correct_port_assignment` | `port_saturation` | Car 1 assigned to port 1 |
| `test_rejected_car_excluded` | `port_saturation` | Cars 2 and 3 in `rejected`; car 1 in `assignments` |
| `test_equal_alloc_charges_fully` | `single_car` | Equal allocation reaches `s_target` within 1e-2 |

Both `run_rolling_horizon` and `run_equal_allocation` are tested where applicable.

---

## Invariant Tests (`test_invariants.py`)

Physical and accounting constraints that must hold across all scenarios and both runners. Tests are parametrized over all three fixtures × both runners.

| Test | What's checked |
|---|---|
| `test_soc_never_exceeds_capacity` | `soc_trajectory[i][t] <= s_cap[i] + 1e-2` for all cars and steps |
| `test_soc_never_negative` | `soc_trajectory[i][t] >= -1e-2` for all cars and steps |
| `test_grid_cap_never_violated` | `sum(I_charge[j,t] * V[j]) <= P_max + 1e-2` for all steps |
| `test_profit_identity` | `abs(net_profit - (revenue - cost)) < 1e-6` |
| `test_no_charge_after_departure` | Current applied at port is 0 for all steps after car departs |
| `test_rejected_cars_have_no_port` | Cars in `rejected` do not appear in `assignments` |

---

## Edge Case Tests (`test_edge_cases.py`)

Each test defines its own minimal scenario locally. Both runners are exercised unless the case is runner-specific.

| Test | Setup | What's checked |
|---|---|---|
| `test_all_ports_full` | J=1, 3 cars arrive at t=1 | Cars 2 and 3 rejected; car 1 served |
| `test_car_already_at_target` | `s_init == s_target` | Car departs at t=1; no energy charged |
| `test_horizon_longer_than_remaining` | H=10, T=3 | No crash; results valid |
| `test_single_timestep` | T=1 | Both runners complete without error |
| `test_car_arrives_at_last_step` | `arr[i] = T` | Car assigned a port; SoC unchanged |
| `test_no_cars` | I=0 | Both runners return empty results without error |
