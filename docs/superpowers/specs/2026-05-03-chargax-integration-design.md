# Chargax Integration Design

**Date:** 2026-05-03
**Status:** Approved, pending implementation

## Context

This document specifies how the external JAX-based EV charging simulator
[Chargax](https://github.com/ponseko/chargax) is integrated with the
existing Pyomo/MILP charging optimisation project. The goal is a
research-grade benchmarking setup where multiple control policies can be
compared fairly inside the same physics simulator.

### Key constraints resolved during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Loop ownership | Chargax owns the step loop | All controllers see identical physics |
| Time resolution | 5-minute steps natively in MILP (`delta_t = 5/60`) | No aggregation needed; MILP is already parameterised by `delta_t` |
| V2G / on-site battery | Disabled for now (`allow_discharging=False`, no battery node) | Scope control; extend later |
| Python version | Python 3.14 with JAX 0.10.0 | JAX 0.10.0 has explicit 3.14 support |
| Chargax installation | `pip install git+https://github.com/ponseko/chargax.git` | Clean, version-pinnable, no code copy |
| Capacity renormalisation | `renormalize_currents=False` | MILP already enforces grid cap; silent rescaling would corrupt comparison |
| Solver | HiGHS by default, Gurobi for publication runs | HiGHS is open-source and bundled with Pyomo |

---

## Architecture

**Architecture 3 — Full Benchmark Harness.** A `BenchmarkRunner` drives
any `BaseController` through a Chargax episode and collects results in a
unified format. The MILP controller is implemented as a stateful rolling
MPC (re-solve every H steps, hold plan between solves).

```
BenchmarkRunner.run_benchmark(controllers, seeds, output_dir)
    └─→ for each (controller, seed):
            BenchmarkRunner.run_episode(controller, seed)
                └─→ env.reset_env(key)
                    while not done:
                        ChargaxWrapper.extract_state(obs, state)   → data dict
                        BaseController.compute_action(state)        → {port_j: amps}
                        ChargaxWrapper.to_chargax_actions(actions)  → MultiDiscrete
                        env.step_env(key, state, actions)           → (timestep, state)
                └─→ ChargaxSimResults
            storage.save(result, path)
```

---

## Folder Structure

Only files marked **NEW** or **FILL IN** require work. Everything under
`optimization/` is untouched.

```
src/evopt/
│
├── env/
│   ├── chargax_wrapper.py        # FILL IN — obs → data dict; actions → Chargax format
│   └── action_mapper.py          # NEW — discretize_amps(amps, i_max, num_levels) → int
│                                 #        standalone so it can be unit-tested independently
│
├── controllers/
│   ├── base_controller.py        # UNCHANGED (add reset() no-op)
│   ├── equal_share.py            # UNCHANGED
│   ├── lp_controller.py          # FILL IN — stateful rolling MPC
│   └── chargax_baselines.py      # NEW — BaseController wrappers for MaxCharge / Random
│
├── benchmarking/                 # NEW directory
│   ├── __init__.py
│   ├── runner.py                 # BenchmarkRunner
│   ├── results.py                # ChargaxSimResults, StepRecord dataclasses
│   └── storage.py                # save / load / build_summary
│
├── experiments/
│   ├── run_tiny_cost_case.py     # UNCHANGED
│   ├── runner.py                 # UNCHANGED
│   ├── station_configs.py        # NEW — Chargax ChargingStation factory
│   └── run_chargax_benchmark.py  # NEW — main entry point
│
├── metrics/
│   └── evaluation.py             # FILL IN — compute_summary_metrics
│
└── tests/
    └── test_chargax_integration.py  # NEW — smoke tests
```

---

## Component Interfaces

### `ChargaxWrapper`

```python
class ChargaxWrapper:
    def __init__(self, station: ChargingStation, num_discretization_levels: int = 10) -> None:
        # extracts J, V, I_max, P_max from the station object at construction time

    def reset(self) -> None:
        # clears _charger_to_car, _prev_connected, _next_car_id

    def extract_state(self, obs: dict, chargax_state) -> dict:
        # Returns the data dict that BaseController.compute_action expects.
        # Keys: t, delta_t, J, P_max, V, I_max, p_buy, p_sell,
        #       present_cars {car_id: {soc_now, s_target, s_cap, t_max}},
        #       assignments {car_id: port_j}

    def to_chargax_actions(self, actions: dict[int, float]) -> dict:
        # actions = {port_j: amps}
        # returns {"evses": jnp.array([level_0, ...]), "batteries": jnp.array([0])}
```

**Internal state tracked between steps:**

```python
self._charger_to_car: dict[int, int]    # {charger_j: car_id}
self._prev_connected: set[int]           # charger indices connected last step
self._next_car_id: int                   # auto-incrementing per episode
```

### `BaseController`

```python
class BaseController:
    def compute_action(self, state: dict) -> dict[int, float]:
        raise NotImplementedError

    def reset(self) -> None:
        pass    # stateless controllers leave this as a no-op
```

### `LPController` (stateful rolling MPC)

```python
class LPController(BaseController):
    def __init__(self, horizon_steps: int = 12, solver: str = "highs") -> None:
        # horizon_steps=12 → 1-hour lookahead at 5-min resolution

    def reset(self) -> None:
        # clears _plan, _step_counter

    def compute_action(self, state: dict) -> dict[int, float]:
        # re-solves if state["t"] % horizon_steps == 0 or _plan is None
        # otherwise returns _plan[state["t"]]
        # delegates to build_rolling_model + solve from existing optimization/
```

### `BenchmarkRunner`

```python
class BenchmarkRunner:
    def __init__(self, env, wrapper: ChargaxWrapper) -> None: ...

    def run_episode(self, controller: BaseController, seed: int) -> ChargaxSimResults: ...

    def run_benchmark(
        self,
        controllers: dict[str, BaseController],
        seeds: list[int],
        output_dir: Path,
    ) -> None: ...
```

### `ChargaxSimResults`

```python
@dataclass
class ChargaxSimResults:
    controller_name: str
    seed: int
    episode_date: str
    net_profit: float
    total_revenue: float
    total_cost: float
    served_customers: int
    rejected_customers: int
    mean_soc_at_departure: float
    step_log: list[StepRecord]

@dataclass
class StepRecord:
    t: int
    actions: dict[int, float]   # {port_j: amps}
    reward: float
    profit_delta: float
    served: int
    rejected: int
```

---

## State Translation

### Chargax EVSE fields → your data dict

| Chargax field | Data dict key | Conversion |
|---|---|---|
| `charger_is_car_connected[j]` | (arrival/departure detection) | flip True → new car; flip False → departure |
| `car_battery_now_kw[j]` | `present_cars[id]["soc_now"]` | direct (kWh) |
| `car_battery_capacity_kw[j]` | `present_cars[id]["s_cap"]` | direct (kWh) |
| `car_desired_battery_percentage[j]` × capacity | `present_cars[id]["s_target"]` | multiply fraction by capacity |
| `car_time_till_leave[j]` / `delta_t_minutes` | `present_cars[id]["t_max"]` | minutes → steps, add current t |
| `max_current[j]` | `I_max[j]` | direct (A) |
| `voltage[j]` | `V[j]` | direct (V) |
| `future_buy_prices[0..H]` | `p_buy[t..t+H]` | repeat each hourly value × 12 steps |
| `future_sell_prices[0..H]` | `p_sell[t..t+H]` | repeat each hourly value × 12 steps |
| station `max_kw_throughput` × 1000 | `P_max` | kW → W |

### Arrival and departure detection

```python
now_connected = {j for j in range(J) if bool(obs["evses"].charger_is_car_connected[j])}

new_arrivals = now_connected - self._prev_connected
departures   = self._prev_connected - now_connected

# Always process departures before arrivals (prevents ID reuse within same step)
for j in departures:
    del self._charger_to_car[j]

for j in new_arrivals:
    self._charger_to_car[j] = self._next_car_id
    self._next_car_id += 1

self._prev_connected = now_connected
```

### Action translation (amps → discrete levels)

```python
def to_chargax_actions(self, actions: dict[int, float]) -> dict:
    levels = []
    for j in range(self.J):
        amps  = actions.get(j, 0.0)
        level = round(amps / self.I_max[j] * self.num_discretization_levels)
        level = int(jnp.clip(level, 0, self.num_discretization_levels))
        levels.append(level)
    return {"evses": jnp.array(levels), "batteries": jnp.array([0])}
```

---

## Simulation Loop (detailed)

```python
def run_episode(self, controller: BaseController, seed: int) -> ChargaxSimResults:
    key = jax.random.PRNGKey(seed)
    obs, state = self.env.reset_env(key)

    controller.reset()
    self.wrapper.reset()

    step_log = []
    done = False

    while not done:
        clean_state    = self.wrapper.extract_state(obs, state)
        actions        = controller.compute_action(clean_state)
        chargax_actions = self.wrapper.to_chargax_actions(actions)

        key, subkey = jax.random.split(key)
        timestep, state = self.env.step_env(subkey, state, chargax_actions)
        obs = timestep.observation

        step_log.append(StepRecord(
            t            = int(state.timestep),
            actions      = actions,
            reward       = float(timestep.reward),
            profit_delta = float(state.profit),
            served       = int(state.served_customers),
            rejected     = int(state.rejected_customers),
        ))

        done = timestep.last()

    return ChargaxSimResults.from_final_state(
        controller_name=controller.__class__.__name__,
        seed=seed,
        state=state,
        step_log=step_log,
    )
```

### Entry point (`run_chargax_benchmark.py`)

```python
from evopt.experiments.station_configs import build_simple_station
from chargax import Chargax

station = build_simple_station(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
env = Chargax(
    station=station,
    minutes_per_timestep=5,
    allow_discharging=False,
    renormalize_currents=False,
)
wrapper = ChargaxWrapper(station)
runner  = BenchmarkRunner(env, wrapper)

runner.run_benchmark(
    controllers={
        "milp_h12":    LPController(horizon_steps=12, solver="highs"),
        "equal_share": EqualShareController(),
        "max_charge":  MaxChargeController(),
        "random":      RandomController(),
    },
    seeds=list(range(10)),
    output_dir=Path("results/chargax"),
)
```

---

## Results Storage

### File layout

```
results/
└── chargax/
    ├── milp_h12/
    │   ├── seed_0.json
    │   └── ...
    ├── equal_share/
    │   └── seed_0.json ...
    ├── max_charge/
    │   └── seed_0.json ...
    ├── random/
    │   └── seed_0.json ...
    └── summary.csv
```

### `storage.py` interface

```python
def save(result: ChargaxSimResults, path: Path) -> None: ...
def load(path: Path) -> ChargaxSimResults: ...
def build_summary(results_dir: Path) -> pd.DataFrame:
    # one row per (controller, seed)
    # columns: controller, seed, net_profit, total_revenue, total_cost,
    #          served_customers, rejected_customers, mean_soc_at_departure, gap_to_best
```

### `metrics/evaluation.py`

```python
def compute_summary_metrics(results: list[ChargaxSimResults]) -> dict:
    # groups by controller_name
    # returns mean ± std across seeds for:
    # net_profit, served_rate, rejection_rate, mean_soc_at_departure
```

---

## Offline Clairvoyant Upper Bound (future addition)

To compute the gap-to-optimal for the paper, run one episode with a
`RecordingController` that logs all arrivals, prices, and departures
without taking any actions. Then solve `build_ev_fcfs_model` on the
collected full-episode scenario. This is a one-liner addition to
`run_chargax_benchmark.py` once the online loop is working.

---

## Risks and Pitfalls

1. **Car identity drift** — if Chargax reuses a charger slot in the same
   step (depart + arrive simultaneously), always process departures before
   arrivals in `extract_state`.

2. **JAX scalar leakage** — every value pulled from `obs["evses"]` is a
   JAX scalar. Wrap all field accesses with `float()` or `int()` before
   passing to Pyomo. Missing one conversion causes silent type errors.

3. **Price lookahead clamping** — Chargax's default `price_hour_lookahead=6`
   gives 72 steps of future prices at 5-min resolution. Clamp price dict
   extraction to `[t, t+H-1]` to match the rolling window.

4. **HiGHS vs Gurobi** — use `solver="highs"` as default in all code
   and tests. Switch to `"gurobi"` only for final publication runs.
   Never hard-code Gurobi in test fixtures.

5. **`renormalize_currents=False`** — must be set on the Chargax env.
   If left True, Chargax silently rescales actions and the MILP output
   is not what gets applied, invalidating the benchmark.

6. **`delta_t` in existing scenarios** — `SAMPLE_DATA` uses `delta_t=1.0`.
   Chargax-driven scenarios use `delta_t=5/60`. The optimisation model
   handles both correctly; the difference only affects scenario data
   construction in `ChargaxWrapper`.
