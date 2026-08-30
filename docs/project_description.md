# EV Charging Optimization — Project Description for Thesis Writing

This document provides a complete description of the repository structure, research objectives,
methodology, implementation decisions, and modeling conventions. It is intended as a
self-contained reference for writing a thesis chapter about this work.

---

## 1. Research Context and Motivation

### 1.1 Problem Setting

Electric vehicle (EV) charging stations operate under a hard grid power cap imposed by the
distribution network operator. This cap limits the total power the station can simultaneously
draw from the grid, creating a resource-allocation problem: the station must decide, at each
5-minute timestep, how much power to send to each charging port, given that not all cars can
be charged at full rate simultaneously.

The station operator earns revenue by selling electricity to EV customers at a fixed or
dynamic tariff, and pays market prices (spot prices) to procure that electricity from the
grid. Profit maximisation requires minimising grid procurement cost while maximising energy
delivered to customers before their departure deadlines.

Two additional factors complicate the problem:

1. **Stochastic arrivals and departures.** Cars arrive and depart randomly throughout the day.
   The controller has no advance knowledge of future arrivals. Departure times are observed
   through the simulator but are estimates (based on how long each customer intended to stay),
   not exact.

2. **On-site Battery Energy Storage System (BESS).** A stationary battery can absorb cheap
   grid electricity during low-price periods and release it during expensive periods, reducing
   net procurement cost through energy arbitrage. The BESS competes with EV charging for
   the limited grid capacity.

### 1.2 Research Role of this Repository

This codebase implements a **receding-horizon Linear Programming (LP) controller** as a
stable, interpretable baseline for the EV charging control problem. The LP is intended to be
compared against Reinforcement Learning (RL) approaches that learn to optimise the same
objective through interaction. The LP provides an upper bound on what a model-based
optimisation approach can achieve with perfect knowledge of the current system state but
without future arrival information.

**The LP is not a MILP (Mixed-Integer Linear Program).** Despite the repository name, all
integer decisions (port assignment, car occupancy) are resolved externally by the Chargax
simulator. The LP optimises only continuous variables: charging currents at each port.

### 1.3 Simulator: Chargax

All experiments use **Chargax** (open-source), a JAX-accelerated discrete-time EV charging
station simulator. Chargax handles:
- Stochastic car arrival and departure processes
- Per-car battery dynamics (SoC tracking, desired charge level, time remaining)
- A configurable grid power cap
- An optional on-site stationary battery
- Discrete action spaces for charger levels (MultiDiscrete)
- Electricity buy/sell price time series

The LP controller interfaces with Chargax through a wrapper (`ChargaxWrapper`) that
translates Chargax observations into a state dictionary suitable for LP construction, and
translates LP-computed continuous currents back into discrete Chargax action levels.

---

## 2. Repository Structure

```
MILP_EV_Optimization/
│
├── src/evopt/                        — main Python package
│   │
│   ├── optimization/                 — LP model construction
│   │   ├── model.py                  — build_ev_lp_model (offline), build_rolling_model (MPC)
│   │   ├── objective.py              — profit maximisation objective with BESS efficiency
│   │   ├── constraints.py            — constraint sets C1–C10 (offline + rolling variants)
│   │   ├── solver.py                 — thin Pyomo/HiGHS wrapper with time-limit guard
│   │   └── variables.py              — legacy variable declarations (unused in MPC path)
│   │
│   ├── controllers/                  — controller implementations
│   │   ├── base_controller.py        — BaseController ABC (compute_action, reset)
│   │   ├── lp_controller.py          — LPController: rolling MPC, re-solves every step
│   │   ├── equal_share.py            — EqualShareController: rule-based baseline
│   │   └── chargax_baselines.py      — MaxChargeController, RandomController
│   │
│   ├── env/                          — Chargax environment adapters
│   │   ├── chargax_wrapper.py        — obs → LP state dict; LP actions → Chargax levels
│   │   └── action_mapper.py          — discretize_amps: continuous current → discrete level
│   │
│   ├── benchmarking/                 — experiment infrastructure
│   │   ├── runner.py                 — BenchmarkRunner: single episode, records step metrics
│   │   ├── results.py                — ChargaxSimResults dataclass, StepRecord
│   │   └── storage.py                — JSON save/load, DataFrame builder
│   │
│   ├── experiments/                  — runnable scripts
│   │   ├── run_chargax_benchmark.py  — main benchmark sweep (ports, horizons, seeds)
│   │   ├── run_horizon_comparison.py — horizon sensitivity table
│   │   ├── run_tiny_cost_case.py     — small offline reference scenario
│   │   ├── run_clairvoyant_benchmark.py — offline LP vs MPC comparison
│   │   └── station_configs.py        — Chargax station factory functions
│   │
│   ├── analysis/                     — post-processing and visualisation
│   │   ├── __main__.py               — CLI entry point (python -m evopt.analysis)
│   │   ├── loader.py                 — CSV loader, cleaning, gap_to_best correction
│   │   ├── utils.py                  — CI computation, horizon extraction, metadata
│   │   ├── stats.py                  — paired t-test, Wilcoxon, paired Cohen's d
│   │   ├── tables.py                 — CSV + LaTeX export with CI columns
│   │   ├── plots.py                  — all chart functions (profit, compute, scaling, radar)
│   │   └── correlation.py            — Pearson correlation matrix, CSV, MD summary
│   │
│   └── metrics/
│       └── evaluation.py             — compute_summary_metrics: mean ± std across seeds
│
├── tests/                            — canonical test suite (115 tests)
│   ├── test_lp_model.py              — offline LP model and constraint tests
│   ├── test_lp_controller.py         — rolling MPC controller tests
│   ├── test_chargax_wrapper.py       — state extraction and action mapping
│   ├── test_runner.py                — benchmark runner integration tests
│   ├── test_results_display.py       — analysis pipeline tests
│   └── …                            — additional constraint and metric tests
│
├── results/                          — experiment outputs (not committed to main history)
│   ├── fixed_tariff_30.csv           — 30-seed benchmark, fixed 0.75 €/kWh tariff
│   ├── dynamic_tariff_30.csv         — 30-seed benchmark, dynamic 1.5× tariff
│   ├── fixed_tariff_30_analysis/     — statistical analysis and charts
│   ├── dynamic_tariff_30_analysis/   — statistical analysis and charts
│   └── metadata.json                 — CLI command + git commit for dynamic experiment
│
└── docs/                             — design documents and specifications
    ├── design.md                     — system architecture overview
    ├── experiments.md                — experiment phases
    └── superpowers/specs/            — detailed feature design specifications
```

---

## 3. Objectives

The repository has four interrelated research objectives:

1. **LP as interpretable baseline.** Implement a receding-horizon LP controller that
   makes provably optimal decisions given the current state and a finite lookahead
   window, serving as a ceiling for rule-based methods and a reference for RL agents.

2. **Horizon sensitivity study.** Quantify how much planning lookahead (H = 1, 3, 6, 12
   steps) improves performance. Specifically: does a 60-minute lookahead significantly
   outperform greedy (1-step) charging?

3. **Scalability study.** Evaluate how controller performance and solve time scale with
   the number of charging ports (3, 6, 12 ports), establishing whether the LP remains
   tractable at realistic station sizes.

4. **Tariff structure comparison.** Assess how the revenue structure (fixed tariff vs.
   dynamic cost-plus pricing) affects the LP's absolute advantage over simpler baselines.

---

## 4. Optimization Model

### 4.1 Decision Variables

All decision variables are continuous and non-negative.

| Variable | Domain | Meaning |
|----------|--------|---------|
| `I_ev[j, t]` | ≥ 0 (A) | Charging current at EV port j, timestep t |
| `I_bess_ch[t]` | ≥ 0 (A) | BESS charging current at timestep t |
| `I_bess_dis[t]` | ≥ 0 (A) | BESS discharging current at timestep t |
| `SoCB[t]` | ≥ 0 (kWh) | BESS state of charge at end of timestep t |
| `soc_car[i, t]` | ≥ 0 (kWh) | Car i state of charge at end of timestep t |

### 4.2 Objective Function

The controller maximises net profit over the planning horizon:

```
Maximise:  ev_revenue − grid_cost

ev_revenue = Σ_{j,t} V_j · I_ev[j,t] · z[j,t] · p_eff[t] · Δt/1000

grid_cost  = Σ_t  [ Σ_j V_j · I_ev[j,t]
                    + V_bess · I_bess_ch[t] / η
                    − V_bess · I_bess_dis[t] · η ]
                  · p_buy[t] · Δt/1000
```

where:
- `V_j` = port voltage (400 V for all EV ports)
- `z[j,t]` = port occupancy (1 if a car is present, 0 otherwise; computed externally from assignments)
- `p_eff[t]` = effective sell price with urgency tiebreak: `p_sell[t] + ε·(t_last − t + 1)/H`
- `ε = 1×10⁻⁴` (small urgency term that breaks ties by preferring earlier charging over later deferral)
- `p_buy[t]` = grid electricity spot buy price at timestep t (€/kWh)
- `p_sell[t]` = EV customer tariff at timestep t (€/kWh)
- `η = 0.95` = BESS one-way efficiency (round-trip ≈ 90%)
- `Δt` = timestep duration in hours (5 min = 5/60 hours)

**BESS efficiency model:** η is applied only in the grid-cost objective term.
Charging draws `I_bess_ch/η` from the grid (efficiency loss on intake); discharging
saves `I_bess_dis·η` from the grid (efficiency loss on output). The SoC dynamics
(C4) use raw commanded current without η, matching Chargax's internal battery update rule.

**BESS arbitrage incentive:** BESS discharge physically reduces the station's net grid draw
rather than selling energy to the external grid. It is therefore priced at the avoided
buy cost (`p_buy`), not a separate sell price. The arbitrage signal is:
`(p_buy_expensive − p_buy_cheap) × η²` — the LP charges the BESS when electricity is
cheap and discharges when it is expensive.

**Urgency tiebreak:** Without the ε term, the LP is indifferent between charging a car at
step 1 or step H when the price is constant. The tiebreak ensures earlier charging is
preferred, which matters in rolling-horizon MPC because a plan deferred to step H is
re-planned at the next solve and never executed.

### 4.3 Constraints

| Label | Mathematical Expression | Interpretation |
|-------|------------------------|----------------|
| C1 | `I_ev[j,t] ≤ I_max[j]` | Port hardware current limit (32 A) |
| C2 | `I_bess_ch[t] ≤ r_ch[t]·I_high`, `I_bess_dis[t] ≤ r_dis[t]·I_low` | SoC-dependent BESS current limits |
| C3 | `Σ_j V_j·I_ev[j,t] + V_bess·(I_bess_ch[t]−I_bess_dis[t]) + L[t] ≤ P_max` | Grid power cap |
| C4 | `SoCB[t] = SoCB[t−1] + (I_bess_ch[t]−I_bess_dis[t])·V_bess·Δt/1000` | BESS SoC dynamics |
| C5 | `SoCB_min ≤ SoCB[t] ≤ SoCB_max` | BESS SoC operating range |
| C6 | `soc_car[i,t] = soc_car[i,t−1] + I_ev[j,t]·V_j·Δt/1000` | Car battery SoC dynamics |
| C7 | `s_min[i] ≤ soc_car[i,t] ≤ s_cap[i]` | Car battery operating range |
| C8 | `I_ev[j,t]·V_j ≤ r_car[i,t]·P_car_max[i]` | SoC-dependent CC/CV charging limit |
| C9 | `I_ev[j,t] = 0` when `z[j,t] = 0` | No current on empty ports |
| C10 | Must-serve pacing floor + hard deadline | Prevents LP procrastination; ensures departure target is met |

**C3 note:** Background load `L[t]` is set to zero (no building load data from Chargax).
This makes the grid-cap constraint slightly optimistic; the wrapper's `to_chargax_actions`
applies a scale-down factor to any action set that would exceed P_max in the simulator.

**C10 must-serve detail:** Two sub-constraints are added per car:
- *Pacing floor*: At every re-solve, car i must charge at least `needed/steps_until_dep` kWh
  this step, where `needed = s_target − soc_now`. This accumulates to the target at
  departure, preventing the LP from repeatedly deferring charging to the end of its window.
  The pacing rate is additionally capped at the grid share budget and port hardware limit.
- *Hard deadline*: If car i departs within the current planning window (`dep_i ≤ t_start + H − 1`),
  the LP is required to deliver the full target: `soc_car[i, dep_i] ≥ s_target[i]`.
  This constraint is skipped when the target is physically unachievable (insufficient time
  or grid capacity) to avoid infeasibility.

If the must-serve constraints make the LP infeasible (rare, occurs in tight grid-cap
scenarios with many nearly-full cars), the controller rebuilds and re-solves without
them (`bare=True` mode) as a fallback so no timestep is ever skipped.

### 4.4 Model Parameterisation

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Port voltage V_j | 400 | V | All EV ports |
| V_bess | 400 | V | BESS nominal voltage |
| I_max per port | 32 | A | Max port current; max power 12.8 kW |
| Grid cap P_max | n_ports × 6,000 | W | Scales with port count |
| BESS capacity | 30 | kWh | |
| BESS I_high = I_low | 25 | A | Max BESS current → 10 kW at 400 V |
| BESS η | 0.95 | — | One-way; round-trip ≈ 90% |
| BESS SoC_min | 3 | kWh | 10% of capacity; battery health floor |
| BESS SoC_max | 30 | kWh | Full capacity |
| Timestep Δt | 5/60 | hours | 5-minute intervals |
| Episode length | 288 | steps | One full day (24 h × 12 steps/h) |

**Grid cap rationale:** With P_max = n_ports × 6 kW, the cap is set at approximately
47% of the simultaneous full-rate draw (`n_ports × I_max × V = n_ports × 12.8 kW`).
The cap is genuinely binding: the LP cannot simply charge all cars at full rate.

---

## 5. Model Predictive Control (Receding Horizon)

### 5.1 MPC Loop

The LP is implemented as a **receding-horizon MPC controller**. At every timestep t:

1. Observe the current Chargax state: which cars are present, their current SoC,
   departure estimates, BESS SoC, and the upcoming price schedule.
2. Build and solve the LP over a rolling window of H steps: `[t, t+H−1]`.
3. Execute only the first step of the plan: actions at step t.
4. At step t+1, receive the updated observation and re-solve.

The re-solve at every step allows the controller to track actual car SoC (which may
deviate from the LP's plan due to Chargax's discrete action levels) and respond to
new arrivals and departures.

### 5.2 Horizon Variants

Four horizon lengths are evaluated:

| Name | H (steps) | Lookahead | Description |
|------|-----------|-----------|-------------|
| milp_h1 | 1 | 5 min | Greedy: optimise only the current step |
| milp_h3 | 3 | 15 min | Short lookahead |
| milp_h6 | 6 | 30 min | Medium lookahead |
| milp_h12 | 12 | 60 min | Full hour lookahead (default) |

H=1 reduces the LP to a greedy allocation problem with no temporal reasoning.
H=12 provides full 60-minute lookahead, the maximum tested.

### 5.3 Warm Starting

At each re-solve, the rolling model is warm-started from the current observation:
- Car SoC values are set to `soc_now[i]` from the Chargax observation.
- BESS SoC is set to `socb_now` from the Chargax observation.
- SoC is clamped to `s_target[i]` before use in the LP to prevent infeasibility from
  Chargax's discretisation occasionally overshooting the target.

---

## 6. Controllers Compared

### 6.1 LPController (LP-MPC)

Rolling receding-horizon LP as described in sections 4–5.
Re-solves at every timestep. Supports BESS arbitrage, must-serve constraints, and
variable horizon lengths.

**Files:** `controllers/lp_controller.py`, `optimization/model.py`

### 6.2 EqualShareController

Rule-based baseline. Splits the grid power cap equally across all cars currently
connected. Each car's allocation is additionally capped at:
- Its own `I_max` (port hardware limit)
- The remaining energy needed (avoids overcharging)

No lookahead, no optimisation. Serves as the primary comparison baseline representing
a simple "fair" allocation policy.

**Files:** `controllers/equal_share.py`

### 6.3 MaxChargeController

Always commands `I_max` (maximum current) at every port for every connected car.
Does not respect the grid power cap in its action output; the wrapper's grid-cap
scaler clips actions proportionally when the total draw exceeds P_max.

Represents the greedy "charge as fast as possible" strategy. Near-optimal when the
grid cap is not frequently binding.

**Files:** `controllers/chargax_baselines.py`

### 6.4 RandomController

Samples a uniform random current in `[0, I_max]` at every port each step.
Serves as a lower-bound baseline. Despite its simplicity, it delivers reasonable
performance because cars eventually receive some charging even without coordination.

**Files:** `controllers/chargax_baselines.py`

---

## 7. Tariff Structures

### 7.1 Fixed Tariff (0.75 €/kWh)

The EV customer is charged a constant rate of 0.75 €/kWh regardless of the current
grid spot price. The station's margin on each kWh is `0.75 − p_buy[t]`, which varies
with the spot price. During high-price periods this margin shrinks or can go negative
if `p_buy[t] > 0.75`.

**Benchmark experiment ID:** `0_75_p3_6_12_h1_3_6_12`

### 7.2 Dynamic Tariff (dynamic:1.5)

The EV customer is charged a cost-plus markup: `p_sell[t] = p_buy[t] × 1.5`. The
station always earns a 50% markup over spot prices, regardless of the absolute price
level. The margin per kWh is `p_buy[t] × 0.5`, which is always positive but scales
proportionally with spot prices.

The dynamic tariff compresses absolute profit margins relative to the fixed tariff
when spot prices are low, but maintains a constant relative margin.

**Benchmark experiment ID:** `dynamic_1_5_p3_6_12_h1_3_6_12`

**Why 1.5× markup specifically:** This rate represents a "cost-plus" pricing model
that avoids regulatory price-fixing concerns while maintaining a viable margin.
Higher multipliers would yield larger absolute profits but may not be commercially
realistic.

---

## 8. Implementation Stack

### 8.1 LP Modelling: Pyomo

The LP is constructed using **Pyomo** (Python Optimization Modeling Objects), a
Python-native algebraic modelling language. Pyomo expresses constraints and objectives
symbolically and translates them to the LP/MIP format expected by the underlying solver.

At each MPC step, a new Pyomo `ConcreteModel` is built from scratch with current
parameter values. This avoids solver warm-starting complications but adds Pyomo model
construction overhead (~1–5 ms per step for the sizes tested).

### 8.2 LP Solver: HiGHS

**HiGHS** (High-performance Linear Optimisation Software) is used as the LP solver,
accessed through the `highspy` Python package. HiGHS is an open-source LP/MIP solver
with state-of-the-art simplex and interior point implementations.

The solver is invoked through Pyomo's `SolverFactory("highs")` interface with:
- A 2-second time limit per solve (prevents degenerate simplex instances from stalling
  an episode; normal solves complete in ≤ 55 ms)
- Explicit infeasibility detection that raises `RuntimeError` when HiGHS reports
  infeasible or infeasible-or-unbounded termination, triggering the bare-mode retry

Gurobi is supported as an alternative solver (`solver="gurobi"`) for validation.

### 8.3 Simulation: JAX / Chargax

Chargax is JAX-based (XLA-compiled). The experiment runner calls `env.reset_env(key)`
and `env.step_env(subkey, state, actions)` at each step. JAX random keys are split
deterministically from the seed, ensuring full reproducibility.

The interaction pattern (LP compute on CPU, Chargax step on GPU/CPU via JAX) creates
a serial bottleneck: each step waits for both the LP solve and the JAX step to complete.

### 8.4 Action Discretisation

Chargax uses a `MultiDiscrete` action space with `num_discretization_levels` levels per
port (default: 10 levels). The action mapper converts a continuous LP current (A) to
the nearest discrete level:

```
level = round(amps / I_max × num_levels)   (unidirectional charging)
level ∈ [0, 2×num_levels] centred at num_levels  (bidirectional V2G)
```

This discretisation introduces a small systematic truncation error: the LP plans with
continuous currents, but only discrete multiples of `I_max/num_levels` are actually
delivered. The gap is small (< 5% of I_max per step) but causes the simulator's car SoC
to differ slightly from the LP's planned SoC, which is corrected at each re-solve by
using the observed SoC.

---

## 9. Benchmarking Framework

### 9.1 Episode Structure

One episode corresponds to one full simulated day (288 steps × 5 min = 24 hours).
At each step the runner records:
- Controller compute time (wall-clock ms)
- Step revenue and cost (computed from actions × prices)
- Served and rejected customer counts (from Chargax state)
- Current reward (Chargax's own reward signal)

At episode end, departure SoC fulfillments are aggregated.

### 9.2 Seed-Based Repetition

Each (controller × port configuration × horizon) combination is run for 30 independent
seeds. Each seed uses a different JAX random key for `env.reset_env`, producing different
car arrival times, battery sizes, and desired charge levels. The seed is deterministically
propagated through each Chargax step via `jax.random.split`.

### 9.3 Metrics

| Metric | Description |
|--------|-------------|
| `net_profit` | `total_revenue − total_cost` (€ per episode) |
| `total_revenue` | Energy sold to EVs at ev_tariff (€) |
| `total_cost` | Grid electricity cost including BESS net energy (€) |
| `served_customers` | Cars that arrived and departed (Chargax cumulative counter) |
| `rejected_customers` | Cars turned away because all ports were occupied |
| `mean_soc_fulfillment` | Mean of `min(soc_at_departure / s_target, 1.0)` across all departing cars |
| `gap_to_best` | `net_profit − best_controller_mean_net_profit` for that seed and port count |
| `total_compute_s` | Total LP wall-clock solve time per episode (seconds) |
| `mean_step_ms` | Mean wall-clock time per timestep (milliseconds) |

**Revenue accounting note:** Revenue reflects the energy actually delivered after the
wrapper's grid-cap scale-down, not the LP's planned delivery. When total EV draw exceeds
P_max, all port currents are scaled uniformly downward.

**Rejected customers note:** Rejections occur entirely at the port-assignment level (all
ports occupied), not due to controller decisions. The LP cannot reduce rejections — it only
optimises the charging of cars already assigned to ports.

### 9.4 Storage Format

Results are stored as CSV with one row per (seed × controller) combination and columns for
all metrics. A `metadata.json` captures the exact CLI command, git commit hash, timestamp,
and Python version for full reproducibility.

---

## 10. Analysis Pipeline

The `evopt.analysis` package (run via `python -m evopt.analysis <input.csv>`) produces:

- `table_full.csv` — mean ± std ± 95% CI for all metrics, all controller/port combos
- `table_best.csv` — best controller per port configuration
- `table_significance.csv` — pairwise Welch's t-test p-values and Cohen's d effect sizes
- `table_robustness.csv` — coefficient of variation, IQR, worst/best case per controller
- `table_tradeoff.csv` — profit vs. compute vs. customer satisfaction summary
- `benchmark_report.md` — narrative report combining all tables and figures
- Multiple `.png` figures (profit/compute scaling, violin/box distributions, radar chart,
  scatter trade-off, correlation heatmap)

Statistical testing uses paired Welch's t-tests across the 30 seeds. Effect sizes are
reported as Cohen's d computed on paired differences (each seed provides one observation
per controller pair).

---

## 11. Key Design Decisions and Their Rationale

### 11.1 LP Not MILP

Port assignment (which car goes to which port) is handled by Chargax, not by the LP.
The LP only decides charging currents, making it a pure LP (all continuous variables)
rather than a Mixed-Integer Program. This is intentional: the LP solves in milliseconds,
which is required for real-time MPC at 5-minute intervals.

### 11.2 Re-solve Every Timestep (Not Every H Steps)

The initial design intention was to re-solve every H steps (execute the full plan, then
re-plan). The implementation re-solves at every step. This increases solver calls by H×
but provides better tracking of actual car SoC (which drifts from the plan due to action
discretisation). The extra cost is negligible given HiGHS's sub-100ms solve times.

### 11.3 BESS in Objective, Not SoC Dynamics

The BESS efficiency η = 0.95 is applied only in the objective function's grid-cost term
(charging costs `I_ch/η` from the grid; discharging saves `I_dis×η`). The SoC dynamics
(C4) use raw commanded current without η. This is a deliberate choice to match Chargax's
internal battery update rule, which also applies η only to the economic accounting, not
the physical SoC tracking.

### 11.4 Must-Serve Grid-Share Cap with 1% Slack

The pacing constraint's grid-share budget is calculated as
`P_max × Δt / (n_cars × 1000) × 0.99`. The 1% slack prevents the combined pacing floors
from exactly saturating P_max (which would make the LP infeasible at floating-point
precision). This was added to reduce the frequency of bare-mode fallbacks.

### 11.5 Zero Background Load

Background load `L[t] = 0` at all timesteps. Chargax does not expose a building load
forecast to the controller. This makes the grid-cap constraint slightly optimistic:
the LP plans for the full P_max capacity, but the actual available capacity may be
reduced by unmodelled background loads. In practice, Chargax's station model typically
has no background load, so this is an exact representation.

### 11.6 Tariff Separation: p_buy vs. p_sell

The LP uses two distinct prices:
- `p_buy[t]`: grid electricity spot price (used in grid_cost; varies by hour)
- `p_sell[t]`: EV customer tariff (used in ev_revenue; fixed or dynamic depending on experiment)

The ChargaxWrapper constructs both from Chargax's `future_buy_prices` and
`future_sell_prices` observation fields. In the fixed tariff experiment, `p_sell` is
always 0.75 regardless of Chargax's sell price. In the dynamic experiment, `p_sell =
p_buy × 1.5`.

### 11.7 HiGHS Solver Time Limit (2 seconds)

A 2-second per-solve time limit is enforced for HiGHS. This prevents a rare degenerate
simplex instance (observed once in 30 seeds for milp_h3 at 3 ports, taking ~7750 ms per
step / ~37 minutes per episode) from distorting benchmark results. All normal solves
complete in ≤ 55 ms, making the limit effectively inactive in practice.

---

## 12. Known Limitations and Scope Boundaries

### 12.1 No Future Arrival Forecast

The LP has no knowledge of cars that have not yet arrived. It only plans for currently
docked cars. This is realistic (a charging station cannot know who will arrive) but means
the LP cannot pre-position the BESS for future demand.

### 12.2 Departure Time Uncertainty

Car departure times are estimated from Chargax's `car_time_till_leave` field (how long
the customer intended to stay). These are observed at arrival and do not update if
customers leave early or stay longer than planned. The rolling re-solve mitigates this
by updating the estimate every 5 minutes.

### 12.3 SoC-Dependent Charging Rate Simplification

The LP uses `r_car[i,t] = 1.0` for all cars and timesteps, meaning it assumes constant-
current (CC) charging at all SoC levels. Real EV chargers switch to constant-voltage (CV)
mode as the battery approaches full charge, reducing the charging rate. Chargax models
this, creating a mismatch: the LP overestimates how much energy can be delivered near
full charge. This is partially corrected by warm-starting from observed SoC at each step.

### 12.4 Single BESS

The wrapper assumes exactly one battery storage unit (`obs["batteries"][0]`). Multiple
BESS units are not supported.

### 12.5 No V2G (Vehicle-to-Grid) in Default Configuration

The BESS supports discharging (set via `--allow-bess-discharging`). EV batteries
themselves do not discharge back to the grid in the default configuration (all `I_ev ≥ 0`).
The `--allow-discharging` flag enables bidirectional EV port action levels in the wrapper's
discretisation, but the LP formulation itself does not model negative EV currents.

### 12.6 Benchmark Scope: Interpretable Baseline, Not State-of-the-Art

The LP is not intended to be the best possible controller for this problem. It is an
interpretable baseline that:
- Makes no learning-based assumptions
- Provides a deterministic, auditable decision at every step
- Represents the optimum achievable with current-state knowledge and a finite lookahead

Its purpose is to serve as a comparison point for reinforcement learning agents and
to quantify the value of model-based optimisation relative to simple heuristics.

---

## 13. Reproducibility

### 13.1 Fixed-Tariff Experiment

```bash
python -m evopt.experiments.run_chargax_benchmark \
  --ports 3 6 12 \
  --horizons 1 3 6 12 \
  --seeds 30 \
  --tariff 0.75 \
  --save results/fixed_tariff_30.csv
```

### 13.2 Dynamic-Tariff Experiment

```bash
python -m evopt.experiments.run_chargax_benchmark \
  --ports 3 6 12 \
  --horizons 1 3 6 12 \
  --seeds 30 \
  --tariff dynamic:1.5 \
  --allow-bess-discharging \
  --allow-discharging \
  --save results/dynamic_tariff_30.csv
```

Git commit for dynamic experiment: `b6a4c63`  
Python: 3.13.5 | HiGHS via `highspy` | Pyomo | JAX | Chargax

### 13.3 Running Analysis

```bash
python -m evopt.analysis results/fixed_tariff_30.csv \
  --output results/fixed_tariff_30_analysis/

python -m evopt.analysis results/dynamic_tariff_30.csv \
  --output results/dynamic_tariff_30_analysis/
```

### 13.4 Test Suite

```bash
pytest tests/ -q          # 115 tests, ~60 seconds
```
