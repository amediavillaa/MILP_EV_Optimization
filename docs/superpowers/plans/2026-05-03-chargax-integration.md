# Chargax Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the Chargax JAX simulator as the simulation engine driving all EV charging controllers (MILP, equal-share, max-charge, random) through a shared benchmark harness that saves unified results for paper comparison.

**Architecture:** Chargax owns the step loop. A `ChargaxWrapper` translates Chargax observations into the data-dict format the MILP already expects. An `LPController` re-solves the rolling MILP every H=12 steps (1-hour lookahead) and holds the plan between solves. A `BenchmarkRunner` runs any `BaseController` through a full Chargax episode and serialises results to JSON.

**Tech Stack:** Python 3.14, JAX 0.10.0, Chargax (pip from GitHub), Pyomo, HiGHS solver, pandas, pytest.

---

## File Map

| Status | File | Responsibility |
|---|---|---|
| NEW | `src/evopt/env/action_mapper.py` | `discretize_amps(amps, i_max, num_levels) → int` |
| FILL IN | `src/evopt/env/chargax_wrapper.py` | Chargax obs → data dict; amps → Chargax MultiDiscrete |
| MODIFY | `src/evopt/controllers/base_controller.py` | Add `reset()` no-op |
| MODIFY | `src/evopt/controllers/equal_share.py` | Update to Chargax state-dict format |
| FILL IN | `src/evopt/controllers/lp_controller.py` | Stateful rolling MPC over existing MILP |
| NEW | `src/evopt/controllers/chargax_baselines.py` | `MaxChargeController`, `RandomController` |
| NEW | `src/evopt/benchmarking/__init__.py` | Package marker |
| NEW | `src/evopt/benchmarking/results.py` | `ChargaxSimResults`, `StepRecord` dataclasses |
| NEW | `src/evopt/benchmarking/storage.py` | `save`, `load`, `build_summary` |
| NEW | `src/evopt/benchmarking/runner.py` | `BenchmarkRunner` |
| NEW | `src/evopt/experiments/station_configs.py` | `build_simple_station` factory |
| NEW | `src/evopt/experiments/run_chargax_benchmark.py` | Main entry point |
| FILL IN | `src/evopt/metrics/evaluation.py` | `compute_summary_metrics` |
| NEW | `src/evopt/tests/test_action_mapper.py` | Unit tests for discretize_amps |
| NEW | `src/evopt/tests/test_chargax_wrapper.py` | Unit tests for ChargaxWrapper |
| NEW | `src/evopt/tests/test_lp_controller.py` | Unit tests for LPController |
| NEW | `src/evopt/tests/test_storage.py` | Unit tests for storage |
| NEW | `src/evopt/tests/test_chargax_integration.py` | End-to-end smoke test |

---

## Task 1: Install Chargax and add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Open `pyproject.toml`. Replace the current minimal content with:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "evopt"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pyomo>=6.0",
    "pandas>=2.0",
    "jax>=0.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "chargax @ git+https://github.com/ponseko/chargax.git",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Install the updated project with dev dependencies**

```
pip install -e ".[dev]"
```

Expected: installs evopt, chargax, pandas, jax, pytest with no errors.

- [ ] **Step 3: Verify Chargax is importable**

```
python -c "from chargax import Chargax, ChargingStation, EVSE; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```
git add pyproject.toml
git commit -m "chore: add chargax, pandas, jax dependencies"
```

---

## Task 2: `action_mapper.py` — discretize amps to charging levels

**Files:**
- Create: `src/evopt/env/action_mapper.py`
- Create: `src/evopt/tests/test_action_mapper.py`

- [ ] **Step 1: Write the failing tests**

Create `src/evopt/tests/test_action_mapper.py`:

```python
import pytest
from evopt.env.action_mapper import discretize_amps


def test_full_current_maps_to_max_level():
    assert discretize_amps(32.0, 32.0, 10) == 10


def test_zero_current_maps_to_zero():
    assert discretize_amps(0.0, 32.0, 10) == 0


def test_half_current_maps_to_half_level():
    assert discretize_amps(16.0, 32.0, 10) == 5


def test_overcurrent_clamps_to_max_level():
    assert discretize_amps(40.0, 32.0, 10) == 10


def test_negative_current_clamps_to_zero():
    assert discretize_amps(-5.0, 32.0, 10) == 0


def test_rounding_rounds_to_nearest():
    # 17/32 * 10 = 5.3125 → rounds to 5
    assert discretize_amps(17.0, 32.0, 10) == 5


def test_zero_i_max_returns_zero():
    assert discretize_amps(10.0, 0.0, 10) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_action_mapper.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — file doesn't exist yet.

- [ ] **Step 3: Implement `action_mapper.py`**

Create `src/evopt/env/action_mapper.py`:

```python
def discretize_amps(amps: float, i_max: float, num_levels: int) -> int:
    """Convert a continuous current (A) to a discrete charging level [0, num_levels]."""
    if i_max <= 0:
        return 0
    level = round(amps / i_max * num_levels)
    return max(0, min(num_levels, level))
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest src/evopt/tests/test_action_mapper.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```
git add src/evopt/env/action_mapper.py src/evopt/tests/test_action_mapper.py
git commit -m "feat: add action_mapper.discretize_amps"
```

---

## Task 3: `results.py` — ChargaxSimResults and StepRecord dataclasses

**Files:**
- Create: `src/evopt/benchmarking/__init__.py`
- Create: `src/evopt/benchmarking/results.py`

- [ ] **Step 1: Create the benchmarking package**

Create `src/evopt/benchmarking/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing tests**

Create `src/evopt/tests/test_results.py`:

```python
import pytest
from evopt.benchmarking.results import ChargaxSimResults, StepRecord


def _make_step(t: int) -> StepRecord:
    return StepRecord(
        t=t,
        actions={1: 16.0},
        reward=0.5,
        profit_delta=0.1,
        served=0,
        rejected=0,
        step_revenue=0.08,
        step_cost=0.04,
    )


def test_step_record_fields():
    s = _make_step(0)
    assert s.t == 0
    assert s.actions == {1: 16.0}
    assert s.step_revenue == 0.08


def test_chargax_sim_results_construction():
    result = ChargaxSimResults(
        controller_name="test",
        seed=0,
        episode_date="2023-01-01",
        net_profit=5.0,
        total_revenue=8.0,
        total_cost=3.0,
        served_customers=4,
        rejected_customers=1,
        mean_soc_at_departure=0.85,
        step_log=[_make_step(0), _make_step(1)],
    )
    assert result.net_profit == 5.0
    assert len(result.step_log) == 2


class MockFinalState:
    timestep = 287
    profit = 12.5
    served_customers = 6
    rejected_customers = 2
    datetime = "2023-03-15"


def test_from_final_state():
    step_log = [_make_step(i) for i in range(3)]
    result = ChargaxSimResults.from_final_state(
        controller_name="milp_h12",
        seed=3,
        state=MockFinalState(),
        step_log=step_log,
        departures_soc=[45.0, 30.0, 60.0],
    )
    assert result.controller_name == "milp_h12"
    assert result.seed == 3
    assert result.net_profit == 12.5
    assert result.served_customers == 6
    assert result.total_revenue == pytest.approx(3 * 0.08)
    assert result.total_cost == pytest.approx(3 * 0.04)
    assert result.mean_soc_at_departure == pytest.approx(45.0)
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest src/evopt/tests/test_results.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `results.py`**

Create `src/evopt/benchmarking/results.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class StepRecord:
    t: int
    actions: dict           # {port_j: amps}
    reward: float
    profit_delta: float
    served: int
    rejected: int
    step_revenue: float = 0.0
    step_cost: float = 0.0


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
    step_log: list[StepRecord] = field(default_factory=list)

    @classmethod
    def from_final_state(
        cls,
        controller_name: str,
        seed: int,
        state,
        step_log: list[StepRecord],
        departures_soc: list[float],
    ) -> ChargaxSimResults:
        mean_soc = sum(departures_soc) / len(departures_soc) if departures_soc else 0.0
        total_revenue = sum(r.step_revenue for r in step_log)
        total_cost = sum(r.step_cost for r in step_log)
        return cls(
            controller_name=controller_name,
            seed=seed,
            episode_date=str(state.datetime),
            net_profit=float(state.profit),
            total_revenue=total_revenue,
            total_cost=total_cost,
            served_customers=int(state.served_customers),
            rejected_customers=int(state.rejected_customers),
            mean_soc_at_departure=mean_soc,
            step_log=step_log,
        )

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest src/evopt/tests/test_results.py -v
```

Expected: all passed.

- [ ] **Step 6: Commit**

```
git add src/evopt/benchmarking/__init__.py src/evopt/benchmarking/results.py src/evopt/tests/test_results.py
git commit -m "feat: add ChargaxSimResults and StepRecord dataclasses"
```

---

## Task 4: `storage.py` — save, load, build_summary

**Files:**
- Create: `src/evopt/benchmarking/storage.py`
- Create: `src/evopt/tests/test_storage.py`

- [ ] **Step 1: Write the failing tests**

Create `src/evopt/tests/test_storage.py`:

```python
import json
from pathlib import Path
import pytest
from evopt.benchmarking.results import ChargaxSimResults, StepRecord
from evopt.benchmarking import storage


def _make_result(name: str, seed: int, profit: float) -> ChargaxSimResults:
    return ChargaxSimResults(
        controller_name=name,
        seed=seed,
        episode_date="2023-01-01",
        net_profit=profit,
        total_revenue=profit + 2.0,
        total_cost=2.0,
        served_customers=5,
        rejected_customers=1,
        mean_soc_at_departure=0.9,
        step_log=[StepRecord(t=0, actions={1: 16.0}, reward=0.1,
                             profit_delta=0.05, served=0, rejected=0)],
    )


def test_save_and_load_roundtrip(tmp_path):
    result = _make_result("milp_h12", 0, 10.5)
    path = tmp_path / "milp_h12" / "seed_0.json"
    storage.save(result, path)
    loaded = storage.load(path)
    assert loaded.controller_name == "milp_h12"
    assert loaded.net_profit == pytest.approx(10.5)
    assert loaded.seed == 0
    assert len(loaded.step_log) == 1
    assert loaded.step_log[0].t == 0


def test_save_creates_parent_dirs(tmp_path):
    result = _make_result("equal_share", 1, 8.0)
    path = tmp_path / "deep" / "nested" / "seed_1.json"
    storage.save(result, path)
    assert path.exists()


def test_build_summary_returns_dataframe(tmp_path):
    for name, profit in [("milp_h12", 10.0), ("equal_share", 7.0)]:
        for seed in range(2):
            r = _make_result(name, seed, profit + seed * 0.1)
            storage.save(r, tmp_path / name / f"seed_{seed}.json")

    df = storage.build_summary(tmp_path)
    assert set(df["controller"].unique()) == {"milp_h12", "equal_share"}
    assert "net_profit" in df.columns
    assert "gap_to_best" in df.columns


def test_build_summary_empty_dir(tmp_path):
    df = storage.build_summary(tmp_path)
    assert len(df) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_storage.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `storage.py`**

Create `src/evopt/benchmarking/storage.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evopt.benchmarking.results import ChargaxSimResults, StepRecord


def save(result: ChargaxSimResults, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)


def load(path: Path) -> ChargaxSimResults:
    with open(path) as f:
        data = json.load(f)
    step_log = [StepRecord(**s) for s in data.pop("step_log")]
    return ChargaxSimResults(**data, step_log=step_log)


def build_summary(results_dir: Path) -> pd.DataFrame:
    rows = []
    for json_file in sorted(Path(results_dir).glob("**/*.json")):
        result = load(json_file)
        rows.append({
            "controller":           result.controller_name,
            "seed":                 result.seed,
            "net_profit":           result.net_profit,
            "total_revenue":        result.total_revenue,
            "total_cost":           result.total_cost,
            "served_customers":     result.served_customers,
            "rejected_customers":   result.rejected_customers,
            "mean_soc_at_departure": result.mean_soc_at_departure,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    best = df.groupby("controller")["net_profit"].mean().max()
    df["gap_to_best"] = df["net_profit"] - best
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest src/evopt/tests/test_storage.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```
git add src/evopt/benchmarking/storage.py src/evopt/tests/test_storage.py
git commit -m "feat: add benchmarking storage (save/load/build_summary)"
```

---

## Task 5: `station_configs.py` — Chargax station factory

**Files:**
- Create: `src/evopt/experiments/station_configs.py`

> **Note:** The Chargax `EVSE` and `ChargingStation` constructor signatures are based on field names from `_station_layout.py`. If the imports fail with unexpected keyword arguments, inspect the actual constructors with `import inspect; print(inspect.signature(EVSE.__init__))` and adjust accordingly.

- [ ] **Step 1: Write the failing test**

Add to `src/evopt/tests/test_chargax_integration.py` (create it now, more tests added later):

```python
from evopt.experiments.station_configs import build_simple_station
from chargax import ChargingStation


def test_build_simple_station_returns_charging_station():
    station = build_simple_station(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    assert isinstance(station, ChargingStation)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest src/evopt/tests/test_chargax_integration.py::test_build_simple_station_returns_charging_station -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `station_configs.py`**

Create `src/evopt/experiments/station_configs.py`:

```python
from chargax import EVSE, ChargingStation


def build_simple_station(
    n_ports: int,
    v: float,
    i_max: float,
    p_max_kw: float,
) -> ChargingStation:
    """Simple station with identical unidirectional ports and no battery."""
    evse = EVSE(
        num_chargers=n_ports,
        voltage=v,
        max_current=i_max,
        efficiency=1.0,
    )
    return ChargingStation(
        max_kw_throughput=p_max_kw,
        connections=[evse],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest src/evopt/tests/test_chargax_integration.py::test_build_simple_station_returns_charging_station -v
```

Expected: PASS. If it fails with unexpected keyword arguments, inspect the EVSE constructor:

```
python -c "from chargax import EVSE; import inspect; print(inspect.signature(EVSE.__init__))"
```

Adjust the keyword names in `station_configs.py` to match, then re-run.

- [ ] **Step 5: Commit**

```
git add src/evopt/experiments/station_configs.py src/evopt/tests/test_chargax_integration.py
git commit -m "feat: add build_simple_station factory"
```

---

## Task 6: Update `base_controller.py` and `equal_share.py`

**Files:**
- Modify: `src/evopt/controllers/base_controller.py`
- Modify: `src/evopt/controllers/equal_share.py`

- [ ] **Step 1: Write the failing test for EqualShareController with new state format**

Create `src/evopt/tests/test_equal_share_chargax.py`:

```python
from evopt.controllers.equal_share import EqualShareController


def _make_state(n_cars: int, p_max_w: float = 10_000.0) -> dict:
    """Build a minimal Chargax-format state dict with n_cars on sequential ports."""
    present_cars = {
        i + 1: {"soc_now": 10.0, "s_target": 30.0, "s_cap": 60.0, "t_max": 100}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    return {
        "t": 0,
        "delta_t": 5 / 60,
        "J": n_cars,
        "P_max": p_max_w,
        "V":    {j: 400.0 for j in range(1, n_cars + 1)},
        "I_max": {j: 32.0  for j in range(1, n_cars + 1)},
        "p_buy":  {0: 0.20},
        "p_sell": {0: 0.40},
        "present_cars": present_cars,
        "assignments": assignments,
        "departed_socs": [],
    }


def test_no_cars_returns_empty():
    ctrl = EqualShareController()
    state = _make_state(0)
    assert ctrl.compute_action(state) == {}


def test_single_car_gets_all_capacity():
    ctrl = EqualShareController()
    state = _make_state(1, p_max_w=4_000.0)
    actions = ctrl.compute_action(state)
    # P_max=4000W / V=400V = 10A, I_max=32A → not capped
    assert actions[1] == pytest.approx(10.0, abs=0.01)


def test_two_cars_split_equally():
    ctrl = EqualShareController()
    state = _make_state(2, p_max_w=8_000.0)
    actions = ctrl.compute_action(state)
    # P_share = 4000W; I = 4000/400 = 10A per car
    assert actions[1] == pytest.approx(10.0, abs=0.01)
    assert actions[2] == pytest.approx(10.0, abs=0.01)


def test_reset_does_not_raise():
    EqualShareController().reset()


import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_equal_share_chargax.py -v
```

Expected: failures — current `EqualShareController` uses old `active_vehicles`/`site_capacity_kw` keys.

- [ ] **Step 3: Update `base_controller.py`**

Replace the contents of `src/evopt/controllers/base_controller.py`:

```python
class BaseController:
    """Abstract interface for all charging controllers."""

    def compute_action(self, state: dict) -> dict[int, float]:
        """Return {port_j: amps} for the current state dict."""
        raise NotImplementedError("Subclasses must implement compute_action().")

    def reset(self) -> None:
        """Called at the start of each episode. Stateless controllers may leave this as a no-op."""
        pass
```

- [ ] **Step 4: Update `equal_share.py`**

Replace the contents of `src/evopt/controllers/equal_share.py`:

```python
from evopt.controllers.base_controller import BaseController


class EqualShareController(BaseController):
    """Splits available grid capacity equally across all present vehicles."""

    def compute_action(self, state: dict) -> dict[int, float]:
        present_cars = state.get("present_cars", {})
        assignments  = state.get("assignments", {})

        if not present_cars:
            return {}

        N       = len(present_cars)
        P_share = state["P_max"] / N       # Watts per car

        actions = {}
        for car_id, port_j in assignments.items():
            V      = state["V"][port_j]
            I_max  = state["I_max"][port_j]
            car    = present_cars[car_id]

            # Cap by port hardware limit
            I_port = P_share / V
            I_capped = min(I_port, I_max)

            # Cap by energy still needed
            energy_needed_kwh = max(0.0, car["s_target"] - car["soc_now"])
            I_needed = energy_needed_kwh / (state["delta_t"] * V / 1000.0)

            actions[port_j] = min(I_capped, I_needed)

        return actions
```

- [ ] **Step 5: Run new tests to verify they pass**

```
pytest src/evopt/tests/test_equal_share_chargax.py -v
```

Expected: all passed.

- [ ] **Step 6: Verify existing tests still pass**

```
pytest src/evopt/tests/test_correctness.py src/evopt/tests/test_invariants.py src/evopt/tests/test_edge_cases.py -v
```

Expected: all passed (existing tests use `run_equal_allocation` from `runner.py`, not `EqualShareController`).

- [ ] **Step 7: Commit**

```
git add src/evopt/controllers/base_controller.py src/evopt/controllers/equal_share.py src/evopt/tests/test_equal_share_chargax.py
git commit -m "feat: update BaseController.reset() and EqualShareController to Chargax state format"
```

---

## Task 7: `chargax_wrapper.py` — state translation

**Files:**
- Modify: `src/evopt/env/chargax_wrapper.py`
- Create: `src/evopt/tests/test_chargax_wrapper.py`

- [ ] **Step 1: Write the failing tests**

Create `src/evopt/tests/test_chargax_wrapper.py`:

```python
import pytest
from evopt.env.chargax_wrapper import ChargaxWrapper


# ---------------------------------------------------------------------------
# Minimal mock objects — no JAX dependency in tests
# ---------------------------------------------------------------------------

class MockEVSE:
    def __init__(self, n, connected, soc, capacity, desired_pct,
                 time_till_leave, voltage=400.0, max_current=32.0):
        self.charger_is_car_connected = connected
        self.car_battery_now_kw       = soc
        self.car_battery_capacity_kw  = capacity
        self.car_desired_battery_percentage = desired_pct
        self.car_time_till_leave      = time_till_leave
        self.voltage                  = voltage
        self.max_current              = max_current


class MockState:
    def __init__(self, timestep=0, profit=0.0, served=0, rejected=0, dt="2023-01-01"):
        self.timestep         = timestep
        self.profit           = profit
        self.served_customers = served
        self.rejected_customers = rejected
        self.datetime         = dt


def _make_obs(evse, buy_prices=None, sell_prices=None):
    return {
        "evses": evse,
        "future_buy_prices":  buy_prices  or [0.20] * 6,
        "future_sell_prices": sell_prices or [0.40] * 6,
    }


def _make_wrapper(n_ports=2):
    return ChargaxWrapper(
        n_ports=n_ports,
        v=400.0,
        i_max=32.0,
        p_max_kw=20.0,
        num_discretization_levels=10,
        minutes_per_step=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [True, False], [20.0, 0.0], [60.0, 60.0],
                    [0.8, 0.0], [60.0, 0.0])
    w.extract_state(_make_obs(evse), MockState(0))
    w.reset()
    assert w._charger_to_car == {}
    assert w._prev_connected == set()
    assert w._next_car_id == 0


def test_no_cars_returns_empty_present():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [False, False], [0.0, 0.0], [60.0, 60.0],
                    [0.8, 0.8], [60.0, 60.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert state["present_cars"] == {}
    assert state["assignments"] == {}


def test_arrival_assigns_new_car_id():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [True, False], [20.0, 0.0], [60.0, 60.0],
                    [0.8, 0.0], [60.0, 0.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert len(state["present_cars"]) == 1
    car_id = list(state["present_cars"].keys())[0]
    assert state["assignments"][car_id] == 1  # charger 0 → port 1


def test_two_arrivals_get_distinct_ids():
    w = _make_wrapper(2)
    evse = MockEVSE(2, [True, True], [20.0, 15.0], [60.0, 60.0],
                    [0.8, 0.8], [60.0, 60.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert len(state["present_cars"]) == 2
    ids = list(state["present_cars"].keys())
    assert ids[0] != ids[1]


def test_departure_removes_car():
    w = _make_wrapper(2)
    evse_on = MockEVSE(2, [True, False], [20.0, 0.0], [60.0, 60.0],
                       [0.8, 0.0], [60.0, 0.0])
    state1 = w.extract_state(_make_obs(evse_on), MockState(0))
    car_id = list(state1["present_cars"].keys())[0]

    evse_off = MockEVSE(2, [False, False], [0.0, 0.0], [60.0, 60.0],
                        [0.0, 0.0], [0.0, 0.0])
    state2 = w.extract_state(_make_obs(evse_off), MockState(1))
    assert car_id not in state2["present_cars"]
    assert state2["departed_socs"] == [pytest.approx(20.0)]


def test_soc_fields_extracted_correctly():
    w = _make_wrapper(1)
    evse = MockEVSE(1, [True], [25.0], [60.0], [0.9], [30.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    car = list(state["present_cars"].values())[0]
    assert car["soc_now"]  == pytest.approx(25.0)
    assert car["s_cap"]    == pytest.approx(60.0)
    assert car["s_target"] == pytest.approx(54.0)  # 0.9 * 60


def test_p_max_converted_to_watts():
    w = _make_wrapper(1)
    evse = MockEVSE(1, [False], [0.0], [60.0], [0.0], [0.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert state["P_max"] == pytest.approx(20_000.0)  # 20 kW → 20000 W


def test_price_dict_populated():
    w = _make_wrapper(1)
    evse = MockEVSE(1, [False], [0.0], [60.0], [0.0], [0.0])
    state = w.extract_state(_make_obs(evse, buy_prices=[0.20] * 6), MockState(0))
    assert state["p_buy"][0] == pytest.approx(0.20)
    assert 0 in state["p_buy"]


def test_to_chargax_actions_full_charge():
    w = _make_wrapper(2)
    actions = w.to_chargax_actions({1: 32.0, 2: 16.0})
    evse_arr = list(actions["evses"])
    assert evse_arr[0] == 10   # 32/32 * 10 = 10
    assert evse_arr[1] == 5    # 16/32 * 10 = 5


def test_to_chargax_actions_missing_port_defaults_to_zero():
    w = _make_wrapper(2)
    actions = w.to_chargax_actions({1: 32.0})  # port 2 missing
    evse_arr = list(actions["evses"])
    assert evse_arr[1] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_chargax_wrapper.py -v
```

Expected: failures (current stub has no real logic).

- [ ] **Step 3: Implement `chargax_wrapper.py`**

Replace the full contents of `src/evopt/env/chargax_wrapper.py`:

```python
from __future__ import annotations

import jax.numpy as jnp

from evopt.env.action_mapper import discretize_amps


class ChargaxWrapper:
    """
    Translates between Chargax observations and the data-dict format
    that BaseController.compute_action expects.

    Port indexing convention:
        Chargax uses 0-indexed charger slots  (j = 0, 1, ..., J-1).
        The MILP and controllers use 1-indexed port IDs (j = 1, 2, ..., J).
        This wrapper converts between the two: port_j = charger_j + 1.
    """

    def __init__(
        self,
        n_ports: int,
        v: float,
        i_max: float,
        p_max_kw: float,
        num_discretization_levels: int = 10,
        minutes_per_step: int = 5,
    ) -> None:
        self.J                        = n_ports
        self.V    = {j + 1: v     for j in range(n_ports)}
        self.I_max = {j + 1: i_max for j in range(n_ports)}
        self.P_max_w                  = p_max_kw * 1000.0
        self.num_discretization_levels = num_discretization_levels
        self.minutes_per_step         = minutes_per_step

        self._charger_to_car: dict[int, int] = {}
        self._prev_connected: set[int]        = set()
        self._next_car_id: int                = 0

    def reset(self) -> None:
        self._charger_to_car = {}
        self._prev_connected = set()
        self._next_car_id    = 0

    def extract_state(self, obs: dict, chargax_state) -> dict:
        evse = obs["evses"]
        t    = int(chargax_state.timestep)

        # ------------------------------------------------------------------
        # Arrival / departure detection
        # ------------------------------------------------------------------
        now_connected = {
            j for j in range(self.J)
            if bool(evse.charger_is_car_connected[j])
        }
        departures   = self._prev_connected - now_connected
        new_arrivals = now_connected - self._prev_connected

        # Always process departures before arrivals (prevents same-step ID reuse)
        departed_socs: list[float] = []
        for j in sorted(departures):
            departed_socs.append(float(evse.car_battery_now_kw[j]))
            del self._charger_to_car[j]

        for j in sorted(new_arrivals):
            self._charger_to_car[j] = self._next_car_id
            self._next_car_id += 1

        self._prev_connected = now_connected

        # ------------------------------------------------------------------
        # Build present-car dict and assignments
        # ------------------------------------------------------------------
        present_cars: dict = {}
        assignments:  dict = {}
        for j, car_id in self._charger_to_car.items():
            s_cap      = float(evse.car_battery_capacity_kw[j])
            desired_pct = float(evse.car_desired_battery_percentage[j])
            s_target   = desired_pct * s_cap
            time_left_min = float(evse.car_time_till_leave[j])
            t_max      = t + max(1, round(time_left_min / self.minutes_per_step))

            present_cars[car_id] = {
                "soc_now":  float(evse.car_battery_now_kw[j]),
                "s_target": s_target,
                "s_cap":    s_cap,
                "t_max":    t_max,
            }
            assignments[car_id] = j + 1   # 0-indexed → 1-indexed port

        # ------------------------------------------------------------------
        # Price lookahead: each hourly Chargax price repeated for each
        # 5-min step in that hour
        # ------------------------------------------------------------------
        steps_per_hour = 60 // self.minutes_per_step
        future_buy  = [float(p) for p in obs["future_buy_prices"]]
        future_sell = [float(p) for p in obs["future_sell_prices"]]

        p_buy: dict[int, float] = {}
        p_sell: dict[int, float] = {}
        for h, (bp, sp) in enumerate(zip(future_buy, future_sell)):
            for s in range(steps_per_hour):
                step = t + h * steps_per_hour + s
                p_buy[step]  = bp
                p_sell[step] = sp

        return {
            "t":            t,
            "delta_t":      self.minutes_per_step / 60.0,
            "J":            self.J,
            "P_max":        self.P_max_w,
            "V":            self.V,
            "I_max":        self.I_max,
            "p_buy":        p_buy,
            "p_sell":       p_sell,
            "present_cars": present_cars,
            "assignments":  assignments,
            "departed_socs": departed_socs,
        }

    def to_chargax_actions(self, actions: dict[int, float]) -> dict:
        """Convert {port_j (1-indexed): amps} to Chargax MultiDiscrete action dict."""
        levels = []
        for j in range(self.J):
            port_j = j + 1
            amps   = actions.get(port_j, 0.0)
            levels.append(discretize_amps(amps, self.I_max[port_j],
                                          self.num_discretization_levels))
        return {
            "evses":     jnp.array(levels, dtype=jnp.int32),
            "batteries": jnp.array([0],    dtype=jnp.int32),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest src/evopt/tests/test_chargax_wrapper.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```
git add src/evopt/env/chargax_wrapper.py src/evopt/tests/test_chargax_wrapper.py
git commit -m "feat: implement ChargaxWrapper (state translation)"
```

---

## Task 8: `lp_controller.py` — stateful rolling MPC

**Files:**
- Modify: `src/evopt/controllers/lp_controller.py`
- Create: `src/evopt/tests/test_lp_controller.py`

- [ ] **Step 1: Write the failing tests**

Create `src/evopt/tests/test_lp_controller.py`:

```python
import pytest
from evopt.controllers.lp_controller import LPController


def _make_state(t: int, n_cars: int = 1, p_max_w: float = 10_000.0) -> dict:
    """Minimal Chargax-format state with n_cars, each needing 5 kWh."""
    present_cars = {
        i + 1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 60.0, "t_max": t + 24}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    J = max(n_cars, 1)
    # Provide prices for a generous window around t
    p_buy  = {s: 0.20 for s in range(t, t + 50)}
    p_sell = {s: 0.40 for s in range(t, t + 50)}
    return {
        "t":            t,
        "delta_t":      5 / 60,
        "J":            J,
        "P_max":        p_max_w,
        "V":            {j: 400.0 for j in range(1, J + 1)},
        "I_max":        {j: 32.0  for j in range(1, J + 1)},
        "p_buy":        p_buy,
        "p_sell":       p_sell,
        "present_cars": present_cars,
        "assignments":  assignments,
        "departed_socs": [],
    }


def test_reset_clears_plan():
    ctrl = LPController(horizon_steps=12, solver="highs")
    ctrl._plan = {0: {1: 10.0}}
    ctrl.reset()
    assert ctrl._plan is None


def test_no_cars_returns_empty():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state = _make_state(0, n_cars=0)
    assert ctrl.compute_action(state) == {}


def test_with_one_car_returns_nonempty_actions():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state = _make_state(0, n_cars=1)
    actions = ctrl.compute_action(state)
    assert 1 in actions
    assert actions[1] >= 0.0


def test_re_solves_at_horizon_boundary():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    first_plan = ctrl._plan

    state12 = _make_state(12)
    ctrl.compute_action(state12)
    second_plan = ctrl._plan

    assert first_plan is not second_plan


def test_holds_plan_between_solves():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    plan_after_t0 = ctrl._plan

    state1 = _make_state(1)
    ctrl.compute_action(state1)
    assert ctrl._plan is plan_after_t0   # same object, not re-solved
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_lp_controller.py -v
```

Expected: failures — stub raises `NotImplementedError`.

- [ ] **Step 3: Implement `lp_controller.py`**

Replace the full contents of `src/evopt/controllers/lp_controller.py`:

```python
from __future__ import annotations

from pyomo.environ import value

from evopt.controllers.base_controller import BaseController
from evopt.optimization.model import build_rolling_model
from evopt.optimization.solver import solve


class LPController(BaseController):
    """
    Stateful rolling MPC controller.

    Re-solves the MILP every `horizon_steps` steps and holds the resulting
    plan for intermediate steps. At each solve, the model looks ahead
    `horizon_steps` time steps (e.g. 12 × 5 min = 1 hour).
    """

    def __init__(
        self,
        horizon_steps: int = 12,
        solver: str = "highs",
        M_big: float = 100.0,
        epsilon: float = 0.1,
        minutes_per_step: int = 5,
    ) -> None:
        self.horizon_steps    = horizon_steps
        self.solver           = solver
        self.M_big            = M_big
        self.epsilon          = epsilon
        self.minutes_per_step = minutes_per_step

        self._plan: dict[int, dict[int, float]] | None = None

    def reset(self) -> None:
        self._plan = None

    def compute_action(self, state: dict) -> dict[int, float]:
        t           = state["t"]
        present     = state["present_cars"]
        assignments = state["assignments"]

        if not present:
            return {}

        if self._plan is None or t % self.horizon_steps == 0:
            self._plan = self._solve(state)

        return self._plan.get(t, {})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_milp_data(self, state: dict) -> dict:
        minutes_per_day = 24 * 60
        T_max  = minutes_per_day // self.minutes_per_step - 1   # 287 for 5-min

        cars = state["present_cars"]
        return {
            "J":        state["J"],
            "T":        T_max,
            "delta_t":  state["delta_t"],
            "P_max":    state["P_max"],
            "M_big":    self.M_big,
            "epsilon":  self.epsilon,
            "V":        state["V"],
            "I_max":    state["I_max"],
            "p_buy":    state["p_buy"],
            "p_sell":   state["p_sell"],
            "s_target": {cid: c["s_target"] for cid, c in cars.items()},
            "s_cap":    {cid: c["s_cap"]    for cid, c in cars.items()},
        }

    def _solve(self, state: dict) -> dict[int, dict[int, float]]:
        t           = state["t"]
        assignments = state["assignments"]
        soc_now     = {cid: c["soc_now"] for cid, c in state["present_cars"].items()}
        data        = self._build_milp_data(state)

        m = build_rolling_model(data, t, self.horizon_steps, assignments, soc_now)
        if m is None:
            return {}

        solve(m, solver=self.solver)

        t_end = min(t + self.horizon_steps - 1, data["T"])
        plan: dict[int, dict[int, float]] = {}
        for step in range(t, t_end + 1):
            step_actions: dict[int, float] = {}
            for j in m.J:
                step_actions[int(j)] = float(value(m.I_charge[j, step]))
            plan[step] = step_actions
        return plan
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest src/evopt/tests/test_lp_controller.py -v
```

Expected: all passed. The `test_with_one_car_returns_nonempty_actions` test requires HiGHS to be installed. Verify with:

```
python -c "from pyomo.environ import SolverFactory; s = SolverFactory('highs'); print(s.available())"
```

If HiGHS is not available, install it: `pip install highspy`.

- [ ] **Step 5: Commit**

```
git add src/evopt/controllers/lp_controller.py src/evopt/tests/test_lp_controller.py
git commit -m "feat: implement LPController stateful rolling MPC"
```

---

## Task 9: `chargax_baselines.py` — MaxCharge and Random controllers

**Files:**
- Create: `src/evopt/controllers/chargax_baselines.py`

- [ ] **Step 1: Write the failing tests**

Create `src/evopt/tests/test_chargax_baselines.py`:

```python
import pytest
from evopt.controllers.chargax_baselines import MaxChargeController, RandomController


def _make_state(n_cars: int = 2) -> dict:
    present_cars = {
        i + 1: {"soc_now": 10.0, "s_target": 30.0, "s_cap": 60.0, "t_max": 100}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    return {
        "t": 0, "delta_t": 5/60, "J": n_cars,
        "P_max": 20_000.0,
        "V":    {j: 400.0 for j in range(1, n_cars + 1)},
        "I_max": {j: 32.0  for j in range(1, n_cars + 1)},
        "p_buy": {0: 0.20}, "p_sell": {0: 0.40},
        "present_cars": present_cars,
        "assignments":  assignments,
        "departed_socs": [],
    }


def test_max_charge_returns_i_max_for_each_port():
    actions = MaxChargeController().compute_action(_make_state(2))
    assert actions[1] == pytest.approx(32.0)
    assert actions[2] == pytest.approx(32.0)


def test_max_charge_empty_state_returns_empty():
    assert MaxChargeController().compute_action(_make_state(0)) == {}


def test_random_returns_non_negative_actions():
    actions = RandomController(seed=0).compute_action(_make_state(2))
    assert all(v >= 0 for v in actions.values())


def test_random_respects_i_max():
    actions = RandomController(seed=42).compute_action(_make_state(2))
    assert all(v <= 32.0 for v in actions.values())


def test_both_controllers_reset_without_error():
    MaxChargeController().reset()
    RandomController().reset()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_chargax_baselines.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `chargax_baselines.py`**

Create `src/evopt/controllers/chargax_baselines.py`:

```python
from __future__ import annotations

import numpy as np

from evopt.controllers.base_controller import BaseController


class MaxChargeController(BaseController):
    """Always charges every connected vehicle at its port's maximum current."""

    def compute_action(self, state: dict) -> dict[int, float]:
        return {
            port_j: state["I_max"][port_j]
            for port_j in state["assignments"].values()
        }


class RandomController(BaseController):
    """Charges each vehicle at a uniformly random current in [0, I_max]."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def compute_action(self, state: dict) -> dict[int, float]:
        return {
            port_j: float(self._rng.uniform(0.0, state["I_max"][port_j]))
            for port_j in state["assignments"].values()
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest src/evopt/tests/test_chargax_baselines.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```
git add src/evopt/controllers/chargax_baselines.py src/evopt/tests/test_chargax_baselines.py
git commit -m "feat: add MaxChargeController and RandomController baselines"
```

---

## Task 10: `benchmarking/runner.py` — BenchmarkRunner

**Files:**
- Create: `src/evopt/benchmarking/runner.py`

- [ ] **Step 1: Write the failing integration smoke test**

Add the following to `src/evopt/tests/test_chargax_integration.py` (append to existing file):

```python
import jax
from evopt.benchmarking.runner import BenchmarkRunner
from evopt.benchmarking import storage
from evopt.env.chargax_wrapper import ChargaxWrapper
from evopt.controllers.equal_share import EqualShareController
from evopt.experiments.station_configs import build_simple_station
from chargax import Chargax


def _make_env_and_wrapper():
    station = build_simple_station(n_ports=2, v=400.0, i_max=32.0, p_max_kw=10.0)
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=False,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=2, v=400.0, i_max=32.0, p_max_kw=10.0,
        num_discretization_levels=10, minutes_per_step=5,
    )
    return env, wrapper


def test_benchmark_runner_completes_one_episode():
    env, wrapper = _make_env_and_wrapper()
    runner = BenchmarkRunner(env, wrapper)
    result = runner.run_episode(EqualShareController(), seed=0)

    assert result.controller_name == "EqualShareController"
    assert result.seed == 0
    assert isinstance(result.net_profit, float)
    assert len(result.step_log) == 288   # 24h * 12 steps/h


def test_benchmark_runner_saves_results(tmp_path):
    env, wrapper = _make_env_and_wrapper()
    runner = BenchmarkRunner(env, wrapper)
    runner.run_benchmark(
        controllers={"equal_share": EqualShareController()},
        seeds=[0, 1],
        output_dir=tmp_path,
    )
    files = list(tmp_path.glob("**/*.json"))
    assert len(files) == 2
```

- [ ] **Step 2: Run smoke tests to verify they fail**

```
pytest src/evopt/tests/test_chargax_integration.py::test_benchmark_runner_completes_one_episode -v
```

Expected: `ModuleNotFoundError` for `benchmarking.runner`.

- [ ] **Step 3: Implement `runner.py`**

Create `src/evopt/benchmarking/runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import jax

from evopt.benchmarking import storage
from evopt.benchmarking.results import ChargaxSimResults, StepRecord
from evopt.controllers.base_controller import BaseController
from evopt.env.chargax_wrapper import ChargaxWrapper


class BenchmarkRunner:
    """Runs any BaseController through a Chargax episode and collects results."""

    def __init__(self, env, wrapper: ChargaxWrapper) -> None:
        self.env     = env
        self.wrapper = wrapper

    def run_episode(self, controller: BaseController, seed: int) -> ChargaxSimResults:
        key          = jax.random.PRNGKey(seed)
        obs, state   = self.env.reset_env(key)

        controller.reset()
        self.wrapper.reset()

        step_log:        list[StepRecord] = []
        departures_soc:  list[float]      = []
        done = False

        while not done:
            clean_state = self.wrapper.extract_state(obs, state)
            departures_soc.extend(clean_state.get("departed_socs", []))

            actions         = controller.compute_action(clean_state)
            chargax_actions = self.wrapper.to_chargax_actions(actions)

            # Compute step financials from actions and current prices
            t        = clean_state["t"]
            delta_t  = clean_state["delta_t"]
            step_revenue = 0.0
            step_cost    = 0.0
            for car_id, port_j in clean_state["assignments"].items():
                amps      = actions.get(port_j, 0.0)
                v         = clean_state["V"][port_j]
                energy_kwh = amps * v * delta_t / 1000.0
                step_revenue += energy_kwh * clean_state["p_sell"].get(t, 0.0)
                step_cost    += energy_kwh * clean_state["p_buy"].get(t, 0.0)

            key, subkey = jax.random.split(key)
            timestep, state = self.env.step_env(subkey, state, chargax_actions)
            obs = timestep.observation

            step_log.append(StepRecord(
                t            = t,
                actions      = {k: float(v) for k, v in actions.items()},
                reward       = float(timestep.reward),
                profit_delta = float(state.profit),
                served       = int(state.served_customers),
                rejected     = int(state.rejected_customers),
                step_revenue = step_revenue,
                step_cost    = step_cost,
            ))

            done = bool(timestep.last())

        return ChargaxSimResults.from_final_state(
            controller_name = controller.__class__.__name__,
            seed            = seed,
            state           = state,
            step_log        = step_log,
            departures_soc  = departures_soc,
        )

    def run_benchmark(
        self,
        controllers: dict[str, BaseController],
        seeds: list[int],
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        for name, ctrl in controllers.items():
            for seed in seeds:
                result = self.run_episode(ctrl, seed)
                storage.save(result, output_dir / name / f"seed_{seed}.json")
```

- [ ] **Step 4: Run smoke tests to verify they pass**

```
pytest src/evopt/tests/test_chargax_integration.py -v
```

Expected: all passed. These tests run real Chargax episodes — each may take 5–30 seconds.

> **Note on `timestep.last()`:** If this raises `AttributeError`, check the jaxnasium API: the method might be `timestep.done` (a field) or `timestep.step_type == StepType.LAST`. Adjust the line `done = bool(timestep.last())` accordingly.

- [ ] **Step 5: Commit**

```
git add src/evopt/benchmarking/runner.py src/evopt/tests/test_chargax_integration.py
git commit -m "feat: implement BenchmarkRunner"
```

---

## Task 11: `metrics/evaluation.py` — compute_summary_metrics

**Files:**
- Modify: `src/evopt/metrics/evaluation.py`

- [ ] **Step 1: Write the failing test**

Create `src/evopt/tests/test_evaluation.py`:

```python
import pytest
from evopt.benchmarking.results import ChargaxSimResults, StepRecord
from evopt.metrics.evaluation import compute_summary_metrics


def _r(name, profit, served, rejected):
    return ChargaxSimResults(
        controller_name=name, seed=0, episode_date="2023-01-01",
        net_profit=profit, total_revenue=profit+2, total_cost=2,
        served_customers=served, rejected_customers=rejected,
        mean_soc_at_departure=0.9,
        step_log=[],
    )


def test_groups_by_controller():
    results = [
        _r("milp", 10.0, 8, 1), _r("milp", 12.0, 9, 0),
        _r("equal", 7.0, 6, 2), _r("equal", 8.0, 7, 1),
    ]
    summary = compute_summary_metrics(results)
    assert "milp" in summary
    assert "equal" in summary


def test_computes_mean_profit():
    results = [_r("milp", 10.0, 8, 1), _r("milp", 12.0, 9, 0)]
    summary = compute_summary_metrics(results)
    assert summary["milp"]["net_profit_mean"] == pytest.approx(11.0)


def test_computes_std_profit():
    results = [_r("milp", 10.0, 8, 1), _r("milp", 12.0, 9, 0)]
    summary = compute_summary_metrics(results)
    assert summary["milp"]["net_profit_std"] == pytest.approx(1.0)


def test_computes_served_rate():
    results = [_r("milp", 10.0, 8, 2)]   # 8 served, 2 rejected → rate = 0.8
    summary = compute_summary_metrics(results)
    assert summary["milp"]["served_rate_mean"] == pytest.approx(0.8)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_evaluation.py -v
```

Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `evaluation.py`**

Replace the full contents of `src/evopt/metrics/evaluation.py`:

```python
from __future__ import annotations

import statistics
from evopt.benchmarking.results import ChargaxSimResults


def compute_summary_metrics(results: list[ChargaxSimResults]) -> dict:
    """
    Group results by controller and return mean ± std for key metrics.

    Returns
    -------
    dict[controller_name, dict[metric, value]]
    """
    grouped: dict[str, list[ChargaxSimResults]] = {}
    for r in results:
        grouped.setdefault(r.controller_name, []).append(r)

    summary = {}
    for name, rs in grouped.items():
        profits      = [r.net_profit        for r in rs]
        served_rates = [
            r.served_customers / max(1, r.served_customers + r.rejected_customers)
            for r in rs
        ]
        reject_rates = [
            r.rejected_customers / max(1, r.served_customers + r.rejected_customers)
            for r in rs
        ]
        socs = [r.mean_soc_at_departure for r in rs]

        summary[name] = {
            "net_profit_mean":          statistics.mean(profits),
            "net_profit_std":           statistics.stdev(profits) if len(profits) > 1 else 0.0,
            "served_rate_mean":         statistics.mean(served_rates),
            "served_rate_std":          statistics.stdev(served_rates) if len(served_rates) > 1 else 0.0,
            "rejection_rate_mean":      statistics.mean(reject_rates),
            "rejection_rate_std":       statistics.stdev(reject_rates) if len(reject_rates) > 1 else 0.0,
            "mean_soc_at_departure":    statistics.mean(socs),
        }
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest src/evopt/tests/test_evaluation.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```
git add src/evopt/metrics/evaluation.py src/evopt/tests/test_evaluation.py
git commit -m "feat: implement compute_summary_metrics"
```

---

## Task 12: `run_chargax_benchmark.py` — main entry point

**Files:**
- Create: `src/evopt/experiments/run_chargax_benchmark.py`

- [ ] **Step 1: Implement the entry point**

Create `src/evopt/experiments/run_chargax_benchmark.py`:

```python
"""
run_chargax_benchmark.py — Run all controllers through Chargax and save results.

Usage:
    python -m evopt.experiments.run_chargax_benchmark
    python -m evopt.experiments.run_chargax_benchmark --seeds 5 --output results/chargax
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chargax import Chargax

from evopt.benchmarking.runner import BenchmarkRunner
from evopt.benchmarking.storage import build_summary
from evopt.controllers.chargax_baselines import MaxChargeController, RandomController
from evopt.controllers.equal_share import EqualShareController
from evopt.controllers.lp_controller import LPController
from evopt.env.chargax_wrapper import ChargaxWrapper
from evopt.experiments.station_configs import build_simple_station


def main(n_seeds: int = 10, output_dir: Path = Path("results/chargax")) -> None:
    N_PORTS   = 3
    VOLTAGE   = 400.0
    I_MAX     = 32.0
    P_MAX_KW  = 20.0

    station = build_simple_station(
        n_ports=N_PORTS, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW
    )
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=False,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=N_PORTS, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
        num_discretization_levels=10,
        minutes_per_step=5,
    )

    runner = BenchmarkRunner(env, wrapper)
    runner.run_benchmark(
        controllers={
            "milp_h12":    LPController(horizon_steps=12, solver="highs"),
            "equal_share": EqualShareController(),
            "max_charge":  MaxChargeController(),
            "random":      RandomController(seed=0),
        },
        seeds=list(range(n_seeds)),
        output_dir=output_dir,
    )

    df = build_summary(output_dir)
    print(df.groupby("controller")[["net_profit", "served_customers",
                                    "rejected_customers"]].mean().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int, default=10)
    parser.add_argument("--output", type=str, default="results/chargax")
    args = parser.parse_args()
    main(n_seeds=args.seeds, output_dir=Path(args.output))
```

- [ ] **Step 2: Verify it imports without errors**

```
python -c "import evopt.experiments.run_chargax_benchmark"
```

Expected: no output (clean import).

- [ ] **Step 3: Do a dry run with 1 seed**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 1 --output results/chargax_test
```

Expected: prints a summary table with 4 controllers after ~1–2 minutes. Check that `results/chargax_test/` contains JSON files for each controller.

- [ ] **Step 4: Commit**

```
git add src/evopt/experiments/run_chargax_benchmark.py
git commit -m "feat: add run_chargax_benchmark entry point"
```

---

## Task 13: Final verification — run full test suite

- [ ] **Step 1: Run all tests**

```
pytest src/evopt/tests/ -v
```

Expected: all tests pass. Any failures indicate a regression — fix before proceeding.

- [ ] **Step 2: Run the benchmark with 3 seeds as a final smoke check**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 3 --output results/chargax_smoke
```

Expected: summary table printed, 12 JSON files in `results/chargax_smoke/`.

- [ ] **Step 3: Commit**

```
git add results/  # only if you want to keep sample results; otherwise add to .gitignore
git commit -m "feat: complete Chargax integration — all controllers benchmarked"
```
