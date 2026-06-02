# Presentation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three improvements identified in the presentation design spec that close the most important gaps before presenting: the clairvoyant offline benchmark, realistic BESS derating, and a more-seeds rerun.

**Architecture:** The clairvoyant benchmark uses a two-pass approach — pass 1 runs a null controller to collect the full episode scenario (all arrivals, departures, prices), pass 2 solves the offline LP once with perfect knowledge and replays the pre-computed schedule. The BESS derating is a parameter change in the benchmark runner. The seeds rerun is a documented CLI command.

**Tech Stack:** Pyomo, HiGHS, JAX/Chargax, pandas, pytest

---

## File Map

**New files:**
- `src/evopt/controllers/offline_lp_controller.py` — `OfflineLPController` that replays a pre-computed schedule + `ScenarioCollector` helper
- `src/evopt/experiments/run_clairvoyant_benchmark.py` — two-pass experiment runner
- `src/evopt/tests/test_offline_lp_controller.py` — unit tests

**Modified files:**
- `src/evopt/analysis/plots.py` — add `plot_optimality_gap` chart
- `src/evopt/analysis/report.py` — surface optimality gap table and significance note
- `src/evopt/experiments/run_chargax_benchmark.py` — add `--bess-derating` flag

---

## Task 1: ScenarioCollector — collect full episode scenario in one pass

**Files:**
- Create: `src/evopt/controllers/offline_lp_controller.py`
- Create: `src/evopt/tests/test_offline_lp_controller.py`

**Background:** The offline LP (`build_ev_lp_model`) needs `arr[i]`, `dep[i]`, `s_init[i]`, port assignments, and full-horizon prices for every car that was served in the episode. `ChargaxWrapper.extract_state` only exposes *currently docked* cars. `ScenarioCollector` runs a full episode with a null controller, recording each car's first appearance (arrival step + initial SoC) and last appearance (departure estimate from `t_max`).

- [ ] **Step 1: Write the failing tests**

```python
# src/evopt/tests/test_offline_lp_controller.py
import pytest
from evopt.controllers.offline_lp_controller import OfflineLPController, ScenarioCollector


def _make_state(t: int, cars: dict | None = None) -> dict:
    """Minimal state dict for testing."""
    if cars is None:
        cars = {1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": t + 10}}
    assignments = {car_id: car_id for car_id in cars}
    J = max(cars) if cars else 1
    return {
        "t": t,
        "delta_t": 5 / 60,
        "J": J,
        "P_max": 10_000.0,
        "V": {j: 400.0 for j in range(1, J + 2)},
        "I_max": {j: 32.0 for j in range(1, J + 1)},
        "p_buy":  {s: 0.20 for s in range(t, t + 300)},
        "p_sell": {s: 0.40 for s in range(t, t + 300)},
        "present_cars": cars,
        "assignments": assignments,
        "departed_fulfillments": [],
        "socb_now": 15.0,
    }


def test_collector_records_arrival():
    collector = ScenarioCollector()
    collector.record(_make_state(t=5, cars={1: {"soc_now": 3.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}}))
    scenario = collector.build_scenario()
    assert scenario["arr"][1] == 5
    assert scenario["s_init"][1] == pytest.approx(3.0)


def test_collector_records_departure():
    collector = ScenarioCollector()
    state5 = _make_state(t=5, cars={1: {"soc_now": 3.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}})
    state6 = _make_state(t=6, cars={1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 15}})
    collector.record(state5)
    collector.record(state6)
    scenario = collector.build_scenario()
    # dep should be t_max from last seen step
    assert scenario["dep"][1] == 15


def test_collector_prices_captured_from_first_step():
    collector = ScenarioCollector()
    collector.record(_make_state(t=0))
    scenario = collector.build_scenario()
    assert 0 in scenario["p_buy"]
    assert scenario["p_buy"][0] == pytest.approx(0.20)


def test_offline_controller_replays_schedule():
    schedule = {5: {1: 10.0}, 6: {1: 8.0}, 7: {}}
    ctrl = OfflineLPController(schedule)
    assert ctrl.compute_action(_make_state(t=5)) == {1: 10.0}
    assert ctrl.compute_action(_make_state(t=6)) == {1: 8.0}
    assert ctrl.compute_action(_make_state(t=7)) == {}


def test_offline_controller_missing_step_returns_empty():
    ctrl = OfflineLPController({})
    assert ctrl.compute_action(_make_state(t=99)) == {}
```

- [ ] **Step 2: Run tests — verify they all fail**

```
cd c:\Users\amedi\Desktop\Publication_Work\MILP_EV_Optimization
python -m pytest src/evopt/tests/test_offline_lp_controller.py -v
```

Expected: `ImportError: cannot import name 'OfflineLPController'`

- [ ] **Step 3: Implement `ScenarioCollector` and `OfflineLPController`**

```python
# src/evopt/controllers/offline_lp_controller.py
from __future__ import annotations

from evopt.controllers.base_controller import BaseController


class ScenarioCollector:
    """Records every state snapshot from a collection-pass episode.

    Call `record(state)` at every step with the output of
    ChargaxWrapper.extract_state. Then call `build_scenario()` to get
    the full-episode data dict required by build_ev_lp_model.
    """

    def __init__(self) -> None:
        self._first_seen:  dict[int, dict] = {}   # car_id -> {arr, s_init, port_j, s_cap, s_target}
        self._last_t_max:  dict[int, int]  = {}   # car_id -> most recent t_max estimate
        self._p_buy:       dict[int, float] = {}
        self._p_sell:      dict[int, float] = {}
        self._scenario_meta: dict = {}             # J, P_max, V, I_max, delta_t

    def record(self, state: dict) -> None:
        t = state["t"]

        # Capture station-level metadata from first step
        if not self._scenario_meta:
            self._scenario_meta = {
                "J":       state["J"],
                "P_max":   state["P_max"],
                "V":       dict(state["V"]),
                "I_max":   dict(state["I_max"]),
                "delta_t": state["delta_t"],
            }

        # Accumulate prices (later steps overwrite earlier for the same key,
        # but since future_buy_prices extends forward, union covers all steps)
        self._p_buy.update(state["p_buy"])
        self._p_sell.update(state["p_sell"])

        present = state.get("present_cars", {})
        assignments = state.get("assignments", {})
        for car_id, car in present.items():
            if car_id not in self._first_seen:
                self._first_seen[car_id] = {
                    "arr":    t,
                    "s_init": car["soc_now"],
                    "port_j": assignments.get(car_id, 1),
                    "s_cap":  car["s_cap"],
                    "s_target": car["s_target"],
                }
            self._last_t_max[car_id] = car["t_max"]

    def build_scenario(self) -> dict:
        """Return data dict compatible with build_ev_lp_model."""
        if not self._first_seen:
            return {}

        meta = self._scenario_meta
        cars = sorted(self._first_seen.keys())
        # Re-map car IDs to 1-indexed integers for build_ev_lp_model
        id_map = {old: new for new, old in enumerate(cars, start=1)}

        T_max = max(self._p_buy.keys()) if self._p_buy else 288
        I = len(cars)

        arr         = {id_map[c]: self._first_seen[c]["arr"]     for c in cars}
        dep         = {id_map[c]: self._last_t_max[c]            for c in cars}
        s_init      = {id_map[c]: self._first_seen[c]["s_init"]  for c in cars}
        s_cap       = {id_map[c]: self._first_seen[c]["s_cap"]   for c in cars}
        s_target    = {id_map[c]: self._first_seen[c]["s_target"] for c in cars}
        assignments = {id_map[c]: self._first_seen[c]["port_j"]  for c in cars}

        J      = meta["J"]
        j_bess = J + 1

        # SoC-dependent car charging ratios: flat 1.0 (same as rolling MPC)
        r_car = {
            (id_map[c], t): 1.0
            for c in cars
            for t in range(arr[id_map[c]], dep[id_map[c]] + 1)
        }

        # Use s_target as s_cap in offline LP so it does not over-serve customers
        # (ensures fair comparison with rolling MPC, which stops at s_target)
        s_cap_offline = {id_map[c]: self._first_seen[c]["s_target"] for c in cars}

        P_car_max = {
            id_map[c]: meta["I_max"].get(self._first_seen[c]["port_j"], 32.0)
                       * meta["V"].get(self._first_seen[c]["port_j"], 400.0)
            for c in cars
        }

        return {
            "J":          J,
            "T":          T_max,
            "I":          I,
            "delta_t":    meta["delta_t"],
            "P_max":      meta["P_max"],
            "V":          {**meta["V"], j_bess: meta["V"].get(j_bess, 400.0)},
            "I_max":      meta["I_max"],
            "I_high":     25.0,
            "I_low":      25.0,
            "p_buy":      self._p_buy,
            "p_sell":     self._p_sell,
            "L":          {t: 0.0 for t in range(1, T_max + 1)},
            "assignments": assignments,
            "arr":        arr,
            "dep":        dep,
            "s_init":     s_init,
            "s_cap":      s_cap_offline,
            "s_min":      {id_map[c]: 0.0 for c in cars},
            "s_target":   s_target,
            "P_car_max":  P_car_max,
            "r_car":      r_car,
            "SoCB_init":  0.0,
            "SoCB_min":   3.0,
            "SoCB_max":   30.0,
            "r_bess_ch":  {t: 1.0 for t in range(1, T_max + 1)},
            "r_bess_dis": {t: 1.0 for t in range(1, T_max + 1)},
        }


class OfflineLPController(BaseController):
    """Replays a pre-computed full-horizon LP schedule step by step.

    Build the schedule dict with build_offline_schedule() before creating
    this controller. The schedule maps {t: {port_j: amps}}.
    """

    def __init__(self, schedule: dict[int, dict[int, float]]) -> None:
        self._schedule = schedule

    def reset(self) -> None:
        pass

    def compute_action(self, state: dict) -> dict[int, float]:
        return dict(self._schedule.get(state["t"], {}))
```

- [ ] **Step 4: Run tests — verify all pass**

```
python -m pytest src/evopt/tests/test_offline_lp_controller.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```
git add src/evopt/controllers/offline_lp_controller.py src/evopt/tests/test_offline_lp_controller.py
git commit -m "feat: add ScenarioCollector and OfflineLPController for clairvoyant benchmark"
```

---

## Task 2: `build_offline_schedule` — solve the full-horizon LP from collected scenario

**Files:**
- Modify: `src/evopt/controllers/offline_lp_controller.py`
- Modify: `src/evopt/tests/test_offline_lp_controller.py`

**Background:** Given a scenario dict from `ScenarioCollector.build_scenario()`, build and solve the full-horizon offline LP using `build_ev_lp_model`, then return a schedule dict `{t: {port_j: amps}}` ready for `OfflineLPController`.

- [ ] **Step 1: Add failing test**

Add to `src/evopt/tests/test_offline_lp_controller.py`:

```python
from evopt.controllers.offline_lp_controller import build_offline_schedule


def test_build_offline_schedule_returns_schedule():
    """Offline schedule must cover the car's dwell window and be non-negative."""
    scenario = {
        "J": 1, "T": 10, "I": 1,
        "delta_t": 5 / 60,
        "P_max": 10_000.0,
        "V": {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0},
        "I_high": 25.0, "I_low": 25.0,
        "p_buy":  {t: 0.20 for t in range(1, 11)},
        "p_sell": {t: 0.40 for t in range(1, 11)},
        "L":      {t: 0.0  for t in range(1, 11)},
        "assignments": {1: 1},
        "arr": {1: 1}, "dep": {1: 8},
        "s_init": {1: 2.0}, "s_cap": {1: 10.0},
        "s_min": {1: 0.0}, "s_target": {1: 10.0},
        "P_car_max": {1: 32.0 * 400.0},
        "r_car": {(1, t): 1.0 for t in range(1, 9)},
        "SoCB_init": 0.0, "SoCB_min": 0.0, "SoCB_max": 0.0,
        "r_bess_ch": {t: 0.0 for t in range(1, 11)},
        "r_bess_dis": {t: 0.0 for t in range(1, 11)},
    }
    schedule = build_offline_schedule(scenario, solver="highs")
    # Schedule must exist for every step in planning horizon
    assert isinstance(schedule, dict)
    assert len(schedule) > 0
    # All current values must be non-negative
    for t, actions in schedule.items():
        for port, amps in actions.items():
            assert amps >= -1e-6, f"Negative current at t={t} port={port}: {amps}"
```

- [ ] **Step 2: Run test — verify it fails**

```
python -m pytest src/evopt/tests/test_offline_lp_controller.py::test_build_offline_schedule_returns_schedule -v
```

Expected: `ImportError: cannot import name 'build_offline_schedule'`

- [ ] **Step 3: Implement `build_offline_schedule`**

Add to `src/evopt/controllers/offline_lp_controller.py` (below the imports, add new import; add function after `OfflineLPController`):

Add to imports at top of file:
```python
from evopt.optimization.model import build_ev_lp_model
from evopt.optimization.solver import solve
from pyomo.environ import value
```

Add function after `OfflineLPController`:
```python
def build_offline_schedule(
    scenario:  dict,
    solver:    str   = "highs",
    eta_bess:  float = 0.95,
) -> dict[int, dict[int, float]]:
    """Solve the full-horizon offline LP and return a schedule dict.

    Returns {t: {port_j: amps}} for t = 1 … T.
    Returns {} if the scenario has no cars or the LP is infeasible.
    """
    if not scenario or scenario.get("I", 0) == 0:
        return {}

    m = build_ev_lp_model(scenario, eta_bess=eta_bess)

    try:
        solve(m, solver=solver)
    except Exception:
        return {}

    J     = scenario["J"]
    T     = scenario["T"]
    j_bess = J + 1

    schedule: dict[int, dict[int, float]] = {}
    for t in range(1, T + 1):
        actions: dict[int, float] = {}
        for j in range(1, J + 1):
            actions[j] = max(0.0, float(value(m.I_ev[j, t])))
        # BESS: positive = discharging (LP convention matches OfflineLPController)
        bess_net = float(value(m.I_bess_dis[t])) - float(value(m.I_bess_ch[t]))
        actions[j_bess] = bess_net
        schedule[t] = actions

    return schedule
```

- [ ] **Step 4: Run tests — verify all pass**

```
python -m pytest src/evopt/tests/test_offline_lp_controller.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```
git add src/evopt/controllers/offline_lp_controller.py src/evopt/tests/test_offline_lp_controller.py
git commit -m "feat: add build_offline_schedule for clairvoyant offline LP"
```

---

## Task 3: `run_clairvoyant_benchmark.py` — two-pass experiment runner

**Files:**
- Create: `src/evopt/experiments/run_clairvoyant_benchmark.py`

**Background:** For each (ports, horizon, seed) configuration: pass 1 runs a null controller to collect the full scenario; pass 2 solves the offline LP and builds `OfflineLPController`; pass 3 runs both the offline controller and the best MPC variant (milp_h12) through `BenchmarkRunner.run_episode` with the same seed. Records `net_profit` for each and computes `optimality_gap = (offline - online) / offline`.

- [ ] **Step 1: Create the script**

```python
# src/evopt/experiments/run_clairvoyant_benchmark.py
"""
run_clairvoyant_benchmark.py — Compare offline clairvoyant LP vs rolling MPC.

Usage:
    python -m evopt.experiments.run_clairvoyant_benchmark
    python -m evopt.experiments.run_clairvoyant_benchmark --seeds 10 --ports 3 6 12
    python -m evopt.experiments.run_clairvoyant_benchmark --save results/clairvoyant.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import pandas as pd
from chargax import Chargax

from evopt.benchmarking.runner import BenchmarkRunner
from evopt.controllers.chargax_baselines import MaxChargeController
from evopt.controllers.lp_controller import LPController
from evopt.controllers.offline_lp_controller import (
    OfflineLPController,
    ScenarioCollector,
    build_offline_schedule,
)
from evopt.env.chargax_wrapper import ChargaxWrapper
from evopt.experiments.station_configs import build_station_with_battery

VOLTAGE       = 400.0
I_MAX         = 32.0
KW_PER_PORT   = 6.0
V_BESS        = 400.0
P_BESS_MAX_KW = 10.0
I_BESS        = P_BESS_MAX_KW * 1000.0 / V_BESS  # 25 A
EV_TARIFF     = 0.75


def _collect_scenario(env, wrapper: ChargaxWrapper, seed: int) -> dict:
    """Pass 1: run null controller to collect full episode scenario."""
    key      = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key)
    wrapper.reset()
    collector = ScenarioCollector()

    done = False
    while not done:
        clean_state = wrapper.extract_state(obs, state)
        collector.record(clean_state)

        # Null action: no current to any port
        null_actions = wrapper.to_chargax_actions({})
        key, subkey = jax.random.split(key)
        timestep, state = env.step_env(subkey, state, null_actions)
        obs  = timestep.observation
        done = bool(timestep.terminated) or bool(timestep.truncated)

    return collector.build_scenario()


def _run_one_port(
    n_ports:  int,
    n_seeds:  int,
    horizons: list[int],
) -> list[dict]:
    P_MAX_KW = n_ports * KW_PER_PORT
    station  = build_station_with_battery(
        n_ports=n_ports, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
        batt_capacity_kwh=30.0, batt_max_kw=P_BESS_MAX_KW, batt_efficiency=0.95,
    )
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=False,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=n_ports, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
        num_discretization_levels=10, minutes_per_step=5,
        ev_tariff=EV_TARIFF,
        v_bess=V_BESS, I_high=I_BESS, I_low=I_BESS,
        socb_min=3.0, socb_max=30.0, p_bess_max_kw=P_BESS_MAX_KW,
        allow_bess_discharging=False,
    )
    runner = BenchmarkRunner(env, wrapper)

    records = []
    for seed in range(n_seeds):
        print(f"  ports={n_ports}  seed={seed} — collecting scenario …", end="\r")

        # Pass 1: collect full scenario
        scenario = _collect_scenario(env, wrapper, seed)

        # Pass 2: solve offline LP once
        schedule = build_offline_schedule(scenario, solver="highs")
        offline_ctrl = OfflineLPController(schedule)

        # Pass 3a: replay offline controller (same seed = same episode)
        offline_result = runner.run_episode(offline_ctrl, seed, name="offline_lp")

        # Pass 3b: run each MPC horizon
        for h in horizons:
            mpc_ctrl = LPController(
                horizon_steps=h, solver="highs",
                v_bess=V_BESS, I_high=I_BESS, I_low=I_BESS,
                socb_min=3.0, socb_max=30.0,
                must_serve=True,
            )
            mpc_result = runner.run_episode(mpc_ctrl, seed, name=f"milp_h{h}")
            offline_profit = offline_result.net_profit
            mpc_profit     = mpc_result.net_profit
            gap = (
                (offline_profit - mpc_profit) / abs(offline_profit)
                if abs(offline_profit) > 1e-6
                else 0.0
            )
            records.append({
                "ports":            n_ports,
                "seed":             seed,
                "horizon":          h,
                "offline_profit":   round(offline_profit, 4),
                "mpc_profit":       round(mpc_profit, 4),
                "optimality_gap":   round(gap, 6),
            })

        # Also record max_charge as a reference baseline
        mc_result = runner.run_episode(MaxChargeController(), seed, name="max_charge")
        records.append({
            "ports":            n_ports,
            "seed":             seed,
            "horizon":          None,
            "offline_profit":   round(offline_result.net_profit, 4),
            "mpc_profit":       round(mc_result.net_profit, 4),
            "optimality_gap":   round(
                (offline_result.net_profit - mc_result.net_profit) / abs(offline_result.net_profit)
                if abs(offline_result.net_profit) > 1e-6 else 0.0,
                6
            ),
        })

    print()
    return records


def main(
    n_seeds:  int            = 10,
    horizons: list[int]      = None,
    ports:    list[int]      = None,
    save_path: str | None    = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 3, 6, 12]
    if ports is None:
        ports = [3, 6, 12]

    all_records = []
    for n_ports in ports:
        all_records.extend(_run_one_port(n_ports, n_seeds, horizons))

    df = pd.DataFrame(all_records)

    print("\n=== Optimality gap: (offline_profit - mpc_profit) / offline_profit ===")
    summary = (
        df[df["horizon"].notna()]
        .groupby(["ports", "horizon"])["optimality_gap"]
        .agg(mean="mean", std="std")
        .round(4)
        .reset_index()
    )
    print(summary.to_string(index=False))

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="?", default=10)
    parser.add_argument("--horizons",type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--ports",   type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--save",    type=str, default=None)
    args = parser.parse_args()
    main(
        n_seeds=args.seeds,
        horizons=args.horizons,
        ports=args.ports,
        save_path=args.save,
    )
```

- [ ] **Step 2: Smoke-test with 1 seed and 3 ports (fast)**

```
python -m evopt.experiments.run_clairvoyant_benchmark --seeds 1 --ports 3 --horizons 1 12
```

Expected: prints a table with `optimality_gap` values and no exceptions.

- [ ] **Step 3: Run full clairvoyant benchmark and save results**

```
python -m evopt.experiments.run_clairvoyant_benchmark --seeds 10 --ports 3 6 12 --horizons 1 3 6 12 --save results/clairvoyant.csv
```

Expected: `results/clairvoyant.csv` created. This takes ~10–30 minutes depending on hardware.

- [ ] **Step 4: Commit**

```
git add src/evopt/experiments/run_clairvoyant_benchmark.py results/clairvoyant.csv
git commit -m "feat: add clairvoyant offline LP benchmark and results"
```

---

## Task 4: Plot optimality gap for the presentation

**Files:**
- Modify: `src/evopt/analysis/plots.py`
- Modify: `src/evopt/tests/test_evaluation.py` (add one smoke test)

**Background:** Add `plot_optimality_gap(df, output_dir)` that reads `results/clairvoyant.csv` and shows mean optimality gap (%) by horizon and port size. This is the key new slide: "how far below optimal is each MPC horizon?"

- [ ] **Step 1: Write failing test**

Add to `src/evopt/tests/test_evaluation.py`:

```python
import pandas as pd
import tempfile
from pathlib import Path
from evopt.analysis.plots import plot_optimality_gap


def test_plot_optimality_gap_creates_file():
    df = pd.DataFrame({
        "ports":          [3, 3, 6, 6],
        "horizon":        [1, 12, 1, 12],
        "optimality_gap": [0.05, 0.02, 0.04, 0.01],
        "seed":           [0, 0, 0, 0],
    })
    with tempfile.TemporaryDirectory() as tmp:
        plot_optimality_gap(df, Path(tmp))
        files = list(Path(tmp).glob("*.png")) + list(Path(tmp).glob("*.pdf"))
        assert len(files) >= 1
```

- [ ] **Step 2: Run test — verify it fails**

```
python -m pytest src/evopt/tests/test_evaluation.py::test_plot_optimality_gap_creates_file -v
```

Expected: `ImportError: cannot import name 'plot_optimality_gap'`

- [ ] **Step 3: Implement `plot_optimality_gap`**

Add to `src/evopt/analysis/plots.py` (append after existing functions):

```python
def plot_optimality_gap(df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart: mean optimality gap (%) by MPC horizon, one line per port size.

    optimality_gap = (offline_profit - mpc_profit) / |offline_profit|
    Only rows with a numeric horizon (MPC rows) are included.
    """
    apply_pub_style()
    mpc = df[df["horizon"].notna()].copy()
    if mpc.empty:
        return

    ports_vals = sorted(mpc["ports"].unique())
    palette    = colorblind_palette(len(ports_vals))

    agg = (
        mpc.groupby(["horizon", "ports"])["optimality_gap"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["ci"]      = 1.96 * agg["std"] / np.sqrt(agg["n"])
    agg["gap_pct"] = agg["mean"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    for port, color in zip(ports_vals, palette):
        sub = agg[agg["ports"] == port].sort_values("horizon")
        ax.errorbar(
            sub["horizon"], sub["gap_pct"],
            yerr=sub["ci"] * 100,
            marker="o", label=f"{port} ports", color=color, capsize=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("MPC Horizon (steps)")
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_title("MPC vs Clairvoyant Offline LP")
    ax.legend(title="Ports")
    fig.tight_layout()

    save_figure(fig, output_dir / "optimality_gap.png")
    plt.close(fig)
```

- [ ] **Step 4: Run test — verify it passes**

```
python -m pytest src/evopt/tests/test_evaluation.py::test_plot_optimality_gap_creates_file -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/evopt/analysis/plots.py src/evopt/tests/test_evaluation.py
git commit -m "feat: add plot_optimality_gap for clairvoyant vs MPC comparison"
```

---

## Task 5: BESS efficiency derating — realistic r_bess values

**Files:**
- Modify: `src/evopt/experiments/run_chargax_benchmark.py`

**Background:** Currently `_build_lp_data` in `LPController` sets `r_bess_ch/r_bess_dis = 1.0` (flat). Realistic BESS efficiency curves derate available capacity at high and low SoC. A simple model: r = 0.95 (5% derating to represent average availability loss from SoC-dependent efficiency and temperature effects). Add a `--bess-derating` CLI flag to expose this as an experiment parameter.

- [ ] **Step 1: Add `bess_derating` parameter to `LPController`**

Open `src/evopt/controllers/lp_controller.py`. Find `__init__` and add the parameter:

```python
def __init__(
    self,
    horizon_steps: int = 12,
    solver: str = "highs",
    v_bess: float = 400.0,
    I_high: float = 25.0,
    I_low: float = 25.0,
    socb_min: float = 3.0,
    socb_max: float = 30.0,
    minutes_per_step: int = 5,
    must_serve: bool = True,
    bess_derating: float = 1.0,   # <-- add this line
) -> None:
    self.horizon_steps    = horizon_steps
    self.solver           = solver
    self.v_bess           = v_bess
    self.I_high           = I_high
    self.I_low            = I_low
    self.socb_min         = socb_min
    self.socb_max         = socb_max
    self.minutes_per_step = minutes_per_step
    self.must_serve       = must_serve
    self.bess_derating    = bess_derating   # <-- add this line
```

Then in `_build_lp_data`, replace the two flat-ratio lines:

```python
# Before:
"r_bess_ch": {step: 1.0 for step in window},
"r_bess_dis":{step: 1.0 for step in window},

# After:
"r_bess_ch": {step: self.bess_derating for step in window},
"r_bess_dis":{step: self.bess_derating for step in window},
```

- [ ] **Step 2: Verify existing tests still pass**

```
python -m pytest src/evopt/tests/test_lp_controller.py -v
```

Expected: all 7 tests pass (derating=1.0 is the default, behaviour unchanged).

- [ ] **Step 3: Add `--bess-derating` flag to run_chargax_benchmark.py**

In `run_chargax_benchmark.py`, add to `_run_one_config` signature:

```python
def _run_one_config(
    ...
    bess_derating: float = 1.0,   # add
) -> pd.DataFrame:
```

In the `controllers` dict inside `_run_one_config`, add `bess_derating=bess_derating` to each `LPController(...)` call:

```python
f"milp_h{h}": LPController(
    horizon_steps=h, solver="highs", **lp_bess_kwargs,
    must_serve=must_serve,
    bess_derating=bess_derating,   # <-- add
)
```

Pass `bess_derating` through `main()` and add the CLI argument:

```python
# In main() signature:
def main(
    ...
    bess_derating: float = 1.0,
    ...
) -> None:

# In _run_one_config call inside main():
df = _run_one_config(
    ...
    bess_derating=bess_derating,
)

# In argparse block:
parser.add_argument(
    "--bess-derating", type=float, default=1.0,
    help="BESS charge/discharge capacity derating factor in [0,1] (default: 1.0 = no derating)",
)

# In main() call at bottom:
main(
    ...
    bess_derating=args.bess_derating,
)
```

- [ ] **Step 4: Smoke-test the new flag**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 1 --ports 3 --horizons 12 --bess-derating 0.95
```

Expected: runs without error, prints results table.

- [ ] **Step 5: Run experiment with realistic derating and save**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 10 --ports 3 6 12 --horizons 1 3 6 12 --allow-bess-discharging --bess-derating 0.95 --save results/fixed_discharging_derating95_tariff.csv
```

- [ ] **Step 6: Commit**

```
git add src/evopt/controllers/lp_controller.py src/evopt/experiments/run_chargax_benchmark.py results/fixed_discharging_derating95_tariff.csv
git commit -m "feat: add bess_derating parameter to LPController and benchmark runner"
```

---

## Task 6: More seeds rerun — 30 seeds for stronger statistical claims

**Background:** No code changes needed. Run the existing benchmark with `--seeds 30` and save to new result files. This overwrites nothing; the 10-seed results stay.

- [ ] **Step 1: Rerun fixed-tariff benchmark with 30 seeds**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 30 --ports 3 6 12 --horizons 1 3 6 12 --save results/fixed_tariff_30seeds.csv
```

- [ ] **Step 2: Rerun discharging-tariff benchmark with 30 seeds**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 30 --ports 3 6 12 --horizons 1 3 6 12 --allow-bess-discharging --allow-discharging --save results/fixed_discharging_30seeds.csv
```

- [ ] **Step 3: Run analysis on 30-seed results**

```
python -m evopt.analysis --input results/fixed_tariff_30seeds.csv --output results/fixed_tariff_30seeds_analysis/
```

- [ ] **Step 4: Commit results**

```
git add results/fixed_tariff_30seeds.csv results/fixed_discharging_30seeds.csv results/fixed_tariff_30seeds_analysis/ results/fixed_discharging_30seeds_analysis/
git commit -m "data: add 30-seed experiment results for stronger significance claims"
```

---

## Execution Order

Run tasks in order — each task builds on the previous one:

1. **Task 1** → `ScenarioCollector` + `OfflineLPController`
2. **Task 2** → `build_offline_schedule` (extends Task 1's file)
3. **Task 3** → clairvoyant experiment runner (uses Task 1 + 2)
4. **Task 4** → optimality gap plot (uses Task 3's results)
5. **Task 5** → BESS derating (independent of Tasks 1–4)
6. **Task 6** → more seeds rerun (independent of all code tasks)

Tasks 5 and 6 can be done in parallel with Tasks 3–4 once Task 2 is complete.

---

## Self-Review Notes

- ScenarioCollector uses `id_map` to re-index car IDs to 1…I for `build_ev_lp_model`. This is necessary because Chargax car IDs are cumulative integers that can exceed I.
- `s_cap_offline = s_target` ensures the offline LP does not over-serve customers beyond what the rolling MPC would do, making the comparison fair.
- The two-pass approach relies on JAX determinism: same PRNG key = same episode. This is valid for Chargax.
- BESS `I_high/I_low` in `ScenarioCollector.build_scenario()` are hardcoded to 25.0 A. If the wrapper's `I_high/I_low` differ, pass them as parameters. For now this matches the experiment defaults.
- Task 6 (more seeds) requires significant compute time (~3–6 hours for 30 seeds × 3 ports × 7 controllers × all horizons).
