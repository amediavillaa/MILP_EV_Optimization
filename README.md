# EV Charging Optimization — LP / MPC Baseline

A receding-horizon linear programming controller for EV charging power allocation, evaluated against rule-based baselines in the [Chargax](https://github.com/ponseko/chargax) simulation environment. The LP serves as a stable, interpretable baseline for comparison with reinforcement learning approaches to minimising EV charging station operating costs.

---

## Problem

An EV charging station operates `J` ports under a grid power cap `P_max`. Cars arrive and depart stochastically, each with a battery to fill before departure. An on-site battery storage system (BESS) can absorb cheap grid energy and release it during expensive periods. At every 5-minute timestep the controller must allocate charging currents `I_{j,t}` (A) to each port, jointly maximising revenue from EV charging and minimising the grid electricity cost.

---

## Optimization Model

### Objective (maximise)

```
profit = ev_revenue − grid_cost

ev_revenue = Σ_{j,t} V_j · I_ev_{j,t} · z_{j,t} · p_eff_t · Δt/1000

grid_cost  = Σ_t [ Σ_j V_j·I_ev_{j,t}
                   + V_b · I_bess_ch_t / η
                   − V_b · I_bess_dis_t · η ]
                 · p_buy_t · Δt/1000
```

`p_eff_t = p_sell_t + ε·(T−t+1)/H` adds a small urgency tiebreak so the LP prefers to charge earlier rather than deferring all charging to the last step of the window.

**BESS efficiency model:** η = 0.95 (one-way; round-trip ≈ 0.90) is applied only in the grid-cost objective — charging costs `I_ch/η` from the grid; discharging saves `I_dis·η`. The SoC dynamics use raw commanded current so that the LP's internal SoC tracking matches the Chargax simulator's update rule.

### Constraints

| Label | Description |
|-------|-------------|
| C1    | `I_ev[j,t] ≤ I_max[j]` — port hardware current limit |
| C2    | `I_bess_ch[t] ≤ r_ch[t]·I_high`, `I_bess_dis[t] ≤ r_dis[t]·I_low` — SoC-dependent BESS current limits |
| C3    | `Σ_j V_j·I_ev[j,t] + V_b·(I_ch[t]−I_dis[t]) + L[t] ≤ P_max` — grid power cap, deducting background load |
| C4    | BESS SoC dynamics: `SoCB[t] = SoCB[t−1] + (I_ch[t]−I_dis[t])·V_b·Δt/1000` |
| C5    | `SoCB_min ≤ SoCB[t] ≤ SoCB_max` — BESS SoC bounds |
| C6    | Car SoC dynamics, warm-started from observed SoC at each re-solve |
| C7    | `s_min[i] ≤ soc_car[i,t] ≤ s_cap[i]` — car battery bounds |
| C8    | `I_ev[j,t]·V_j ≤ r_car[i,t]·P_car_max[i]` — SoC-dependent CC/CV charging limit |
| C9    | `I_ev[j,t] = 0` when port is unoccupied |
| C10   | Must-serve: pacing floor at every re-solve + hard deadline for departing cars |

### Receding-horizon MPC

The LP is solved at every timestep over a rolling window of `H` steps. Car SoC and BESS SoC are warm-started from the current observation. If the must-serve constraints make the problem infeasible (rare, tight grid-cap scenarios), the controller retries without them as a fallback.

---

## Repository Layout

```
src/evopt/
├── optimization/       LP model builders
│   ├── model.py        build_ev_lp_model  (offline)  /  build_rolling_model (MPC step)
│   ├── objective.py    profit-maximisation objective with BESS eta model
│   ├── constraints.py  C1–C11 constraint sets for offline and rolling models
│   ├── solver.py       thin Pyomo solver wrapper
│   └── variables.py    legacy MILP variable declarations (unused)
│
├── controllers/        controller interface and implementations
│   ├── base_controller.py   BaseController ABC
│   ├── lp_controller.py     LPController — rolling MPC, re-solves every step
│   ├── equal_share.py       EqualShareController — splits P_max/N equally
│   └── chargax_baselines.py MaxChargeController, RandomController
│
├── env/                Chargax environment adapters
│   ├── chargax_wrapper.py   obs → LP state dict, LP actions → Chargax actions
│   └── action_mapper.py     continuous current → discrete Chargax level
│
├── benchmarking/       experiment infrastructure
│   ├── runner.py       BenchmarkRunner — runs one full episode, records metrics
│   ├── results.py      ChargaxSimResults dataclass + StepRecord
│   └── storage.py      JSON save/load and DataFrame summary builder
│
├── experiments/        runnable scripts
│   ├── run_chargax_benchmark.py   main benchmark (ports, horizons, BESS flags)
│   ├── run_horizon_comparison.py  sweep H=1/3/6/12 and print table
│   ├── run_tiny_cost_case.py      small offline reference scenario
│   └── station_configs.py         Chargax station factory functions
│
└── metrics/
    └── evaluation.py   compute_summary_metrics — mean ± std across seeds

tests/                  canonical test suite (91 tests total)
src/evopt/tests/        legacy test suite (also 91 tests, run by default)
```

---

## Controllers

| Name | Class | Description |
|------|-------|-------------|
| `milp_h{H}` | `LPController` | Rolling MPC LP, horizon H steps, re-solves every timestep |
| `equal_share` | `EqualShareController` | Splits P_max equally across present cars, capped by I_max and remaining need |
| `max_charge` | `MaxChargeController` | Always charges at I_max; ignores grid cap (scaled down by wrapper) |
| `random` | `RandomController` | Uniform random current in [0, I_max] |

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11, Pyomo, HiGHS (via `highspy`), JAX, and Chargax.

HiGHS is the default open-source solver. To use Gurobi, pass `solver="gurobi"` to `LPController`.

---

## Running Experiments

### Main benchmark

```bash
# Default: 3 ports, H=12, BESS enabled, fixed tariff 0.75 €/kWh, 10 seeds
python -m evopt.experiments.run_chargax_benchmark

# Sweep port counts and horizon lengths
python -m evopt.experiments.run_chargax_benchmark --ports 3 6 12 --horizons 1 6 12 24

# Dynamic tariff (LP earns live Chargax p_sell price)
python -m evopt.experiments.run_chargax_benchmark --tariff dynamic

# Cost-plus dynamic pricing (p_buy × 1.3)
python -m evopt.experiments.run_chargax_benchmark --tariff dynamic:1.3

# Disable BESS
python -m evopt.experiments.run_chargax_benchmark --no-bess

# Enable BESS discharge (arbitrage)
python -m evopt.experiments.run_chargax_benchmark --allow-bess-discharging

# Disable must-serve constraints
python -m evopt.experiments.run_chargax_benchmark --no-must-serve
```

### Horizon comparison

```bash
python -m evopt.experiments.run_horizon_comparison
python -m evopt.experiments.run_horizon_comparison --seeds 5 --output results/horizon
```

### Key metrics reported

| Metric | Description |
|--------|-------------|
| `net_profit` | total_revenue − total_cost (€) |
| `total_revenue` | energy sold to EVs at ev_tariff (€) |
| `total_cost` | grid electricity purchased + BESS net energy cost (€) |
| `served_customers` | cars that departed (Chargax counter) |
| `rejected_customers` | cars turned away due to no free port |
| `mean_soc_fulfillment` | mean of min(soc_at_departure / s_target, 1.0) across all departing cars |
| `gap_to_best` | net_profit − best controller's mean net_profit |
| `total_compute_s` | total LP solve time per episode (s) |
| `mean_step_ms` | average time per timestep (ms) |

---

## Tests

```bash
# Run full suite (91 tests in tests/ + 91 in src/evopt/tests/)
pytest

# Canonical tests only
pytest tests/

# A single file
pytest tests/test_grid_cap_load.py -v
```

Both test directories are included in `pyproject.toml`'s `testpaths`.

---

## Configuration defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| Voltage | 400 V | All EV ports and BESS |
| I_max | 32 A | Per EV port (12.8 kW max) |
| Grid cap | n_ports × 6 kW | Scales with port count (~47% utilisation at I_max) |
| BESS capacity | 30 kWh | |
| BESS max power | 10 kW (25 A at 400 V) | |
| BESS efficiency η | 0.95 (one-way) | Round-trip ≈ 0.90 |
| Timestep | 5 min | |
| EV tariff | 0.75 €/kWh | Fixed; pass `--tariff dynamic` for spot price |
| Default horizon | 12 steps (60 min) | |

---

## Research Goals

- LP as interpretable baseline vs. reinforcement learning for cost minimisation
- Scalability study across port counts (3, 6, 12)
- Horizon sensitivity: H=1 (greedy) vs. H=12 (1-hour lookahead) vs. H=24
- BESS arbitrage value at different electricity price spreads
- Must-serve vs. profit trade-off under tight grid constraints
