# BESS Objective & Profit Tracking Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs: (1) BESS discharge mispriced in LP objective at `p_sell_bess` instead of `p_buy`; (2) benchmark runner excludes BESS energy from step-level cost tracking.

**Architecture:** Targeted edits across 5 source files. Fix the objective root cause first (Task 2: `objective.py` + `model.py`), then delete the now-dead `p_sell_bess` parameter downstream (Tasks 3–4), then fix the runner accounting (Task 5). Tests in Tasks 2–5 follow TDD: write failing test → fix code → verify pass → commit.

**Tech Stack:** Python 3.11, Pyomo 6+, HiGHS solver (via `SolverFactory("highs")`), JAX, pytest

---

## File Map

| File | Action |
|------|--------|
| `tests/__init__.py` | Create: empty, makes `tests/` a package |
| `tests/conftest.py` | Create: shared `minimal_bess_data` fixture |
| `tests/test_objective_bess.py` | Create: tests for objective fix |
| `tests/test_lp_data.py` | Create: test for `lp_controller` data dict |
| `tests/test_wrapper_state.py` | Create: test for wrapper state dict |
| `tests/test_runner_bess.py` | Create: tests for runner step cost |
| `src/evopt/optimization/objective.py` | Fix: remove `bess_sell_revenue`; subtract BESS discharge from `grid_cost` |
| `src/evopt/optimization/model.py` | Remove: `m.p_sell_bess` Param from both model builders |
| `src/evopt/controllers/lp_controller.py` | Remove: `"p_sell_bess"` key from `_build_lp_data` |
| `src/evopt/env/chargax_wrapper.py` | Remove: `p_sell_bess` dict construction and state key |
| `src/evopt/benchmarking/runner.py` | Fix: add BESS net energy to `step_cost` after car loop |

---

### Task 1: Bootstrap test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/__init__.py`**

Create an empty file at `tests/__init__.py` (no content needed).

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import pytest


@pytest.fixture
def minimal_bess_data():
    """1-port, 1-car, 2-step offline LP scenario with a pre-charged BESS."""
    J = 1
    j_bess = J + 1
    return {
        "J": J,
        "T": 2,
        "I": 1,
        "delta_t": 1.0,            # 1 hour per step — simplifies energy arithmetic
        "P_max": 20_000.0,         # 20 kW — non-binding
        "V":      {1: 400.0, j_bess: 400.0},
        "I_max":  {1: 32.0},
        "I_high": 25.0,
        "I_low":  25.0,
        "p_buy":  {1: 0.10, 2: 0.30},   # cheap at t=1, expensive at t=2
        "p_sell": {1: 0.50, 2: 0.50},
        "L":      {1: 0.0,  2: 0.0},
        "SoCB_min":  0.0,
        "SoCB_max":  20.0,
        "SoCB_init": 10.0,         # BESS starts half-charged
        "r_bess_ch":  {1: 1.0, 2: 1.0},
        "r_bess_dis": {1: 1.0, 2: 1.0},
        "assignments": {1: 1},
        "arr": {1: 1},
        "dep": {1: 2},
        "s_init":    {1: 0.0},
        "s_cap":     {1: 20.0},
        "s_min":     {1: 0.0},
        "s_target":  {1: 10.0},
        "P_car_max": {1: 12_800.0},    # 400 V × 32 A
        "r_car":     {(1, 1): 1.0, (1, 2): 1.0},
    }
```

- [ ] **Step 3: Verify collection**

Run: `pytest --collect-only`
Expected: `no tests ran`, zero errors

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add test scaffold and minimal_bess_data fixture"
```

---

### Task 2: Fix `objective.py` and `model.py`

**Files:**
- Create: `tests/test_objective_bess.py`
- Modify: `src/evopt/optimization/objective.py` (full rewrite)
- Modify: `src/evopt/optimization/model.py` (two deletions)

- [ ] **Step 1: Write failing tests**

Create `tests/test_objective_bess.py`:

```python
import pytest
from pyomo.environ import value


def test_model_has_no_p_sell_bess(minimal_bess_data):
    """After fix, the LP model should not carry a p_sell_bess parameter."""
    from evopt.optimization.model import build_ev_lp_model

    m = build_ev_lp_model(minimal_bess_data)
    assert not hasattr(m, "p_sell_bess")


def test_bess_discharge_reduces_objective_value(minimal_bess_data):
    """
    Discharging BESS at the expensive step (t=2, p_buy=0.30) should reduce
    the negated-profit objective compared to no discharge.

    Under the corrected grid_cost formula, each kWh discharged at t=2 saves
    0.30 in grid cost → more profit → lower objective (sense=minimize).
    """
    from evopt.optimization.model import build_ev_lp_model

    m = build_ev_lp_model(minimal_bess_data)

    for j in m.J_ev:
        for t in m.T:
            m.I_ev[j, t].fix(10.0)
    for t in m.T:
        m.I_bess_ch[t].fix(0.0)
        m.I_bess_dis[t].fix(0.0)
        m.SoCB[t].fix(5.0)
    for i in m.I:
        for t in m.T:
            m.soc_car[i, t].fix(2.0)

    obj_no_discharge = value(m.obj)

    m.I_bess_dis[2].fix(5.0)    # discharge 5 A at expensive t=2
    obj_with_discharge = value(m.obj)

    # More BESS discharge → less grid cost → more profit → lower negated obj
    assert obj_with_discharge < obj_no_discharge
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_objective_bess.py -v`
Expected:
- `test_model_has_no_p_sell_bess` — FAIL (`m.p_sell_bess` exists)
- `test_bess_discharge_reduces_objective_value` — FAIL (old formula adds `bess_sell_revenue` instead of subtracting discharge from `grid_cost`, so discharge does not lower the objective when `p_sell_bess < p_buy`)

- [ ] **Step 3: Replace `src/evopt/optimization/objective.py`**

```python
"""
objective.py — Profit-maximisation objective for EV charging LP models
=======================================================================
Profit maximisation:

    ev_revenue = Σ_{j,t} V_j · I_ev_{j,t} · z_{j,t} · p_eff_t · delta_t/1000
    grid_cost  = Σ_t (Σ_j V_j·I_ev_{j,t} + V_bess·I_bess_ch_t − V_bess·I_bess_dis_t)
                      · p_buy_t · delta_t/1000
    objective  = maximise (ev_revenue − grid_cost)

    p_eff_t = p_sell_t + ε·(t_last − t + 1)/H  where ε = 1e-4

    BESS discharge reduces the station's net grid draw rather than selling to the
    external grid.  It is therefore priced at the avoided buy cost (p_buy), not a
    separate sell price.  The arbitrage incentive is p_buy_expensive − p_buy_cheap:
    the LP charges the BESS when electricity is cheap and discharges when expensive.
"""

from pyomo.environ import Objective, minimize, value

_URGENCY_EPS = 1e-4


def add_offline_profit_objective(m, j_bess: int) -> None:
    def obj_rule(m):
        steps  = sorted(m.T)
        t_last = steps[-1]
        H      = len(steps)
        ev_revenue = sum(
            m.V[j] * m.I_ev[j, t] * value(m.z[j, t])
            * (m.p_sell[t] + _URGENCY_EPS * (t_last - t + 1) / H)
            * m.delta_t / 1000.0
            for j in m.J_ev for t in m.T
        )
        grid_cost = sum(
            (
                sum(m.V[j] * m.I_ev[j, t] for j in m.J_ev)
                + m.V[j_bess] * m.I_bess_ch[t]
                - m.V[j_bess] * m.I_bess_dis[t]
            )
            * m.p_buy[t] * m.delta_t / 1000.0
            for t in m.T
        )
        return -(ev_revenue - grid_cost)

    m.obj = Objective(rule=obj_rule, sense=minimize)


def add_rolling_profit_objective(m, j_bess: int) -> None:
    def obj_rule(m):
        steps  = sorted(m.WIN)
        t_last = steps[-1]
        H      = len(steps)
        ev_revenue = sum(
            m.V[j] * m.I_ev[j, t] * value(m.z[j, t])
            * (m.p_sell[t] + _URGENCY_EPS * (t_last - t + 1) / H)
            * m.delta_t / 1000.0
            for j in m.J_ev for t in m.WIN
        )
        grid_cost = sum(
            (
                sum(m.V[j] * m.I_ev[j, t] for j in m.J_ev)
                + m.V[j_bess] * m.I_bess_ch[t]
                - m.V[j_bess] * m.I_bess_dis[t]
            )
            * m.p_buy[t] * m.delta_t / 1000.0
            for t in m.WIN
        )
        return -(ev_revenue - grid_cost)

    m.obj = Objective(rule=obj_rule, sense=minimize)
```

- [ ] **Step 4: Remove `m.p_sell_bess` from `build_ev_lp_model` in `model.py`**

In `src/evopt/optimization/model.py`, inside `build_ev_lp_model`, delete this line (currently after `m.p_sell` and before `m.L`):

```python
m.p_sell_bess = Param(m.T, initialize=data.get("p_sell_bess", data["p_sell"]))
```

- [ ] **Step 5: Remove `m.p_sell_bess` from `build_rolling_model` in `model.py`**

In `src/evopt/optimization/model.py`, inside `build_rolling_model`, delete these four lines:

```python
_p_sell_bess = data.get("p_sell_bess", data["p_sell"])
```

```python
m.p_sell_bess = Param(m.WIN, initialize={
    t: _p_sell_bess[t] for t in range(t_start, t_end + 1)
})
```

- [ ] **Step 6: Run tests to confirm they pass**

Run: `pytest tests/test_objective_bess.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add tests/test_objective_bess.py \
        src/evopt/optimization/objective.py \
        src/evopt/optimization/model.py
git commit -m "fix: price BESS discharge at p_buy in LP objective; remove p_sell_bess param from model"
```

---

### Task 3: Remove `p_sell_bess` from `lp_controller.py`

**Files:**
- Create: `tests/test_lp_data.py`
- Modify: `src/evopt/controllers/lp_controller.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_lp_data.py`:

```python
def test_lp_data_has_no_p_sell_bess():
    """_build_lp_data should not include p_sell_bess in the output dict."""
    from evopt.controllers.lp_controller import LPController

    ctrl = LPController(horizon_steps=2)
    state = {
        "t": 1,
        "J": 1,
        "delta_t": 1 / 12,
        "P_max": 6_000.0,
        "V":     {1: 400.0},
        "I_max": {1: 32.0},
        "p_buy":       {t: 0.20 for t in range(1, 300)},
        "p_sell":      {t: 0.50 for t in range(1, 300)},
        "p_sell_bess": {t: 0.15 for t in range(1, 300)},
        "present_cars": {
            1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 20.0, "t_max": 12},
        },
        "assignments": {1: 1},
    }
    data = ctrl._build_lp_data(state)
    assert "p_sell_bess" not in data
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/test_lp_data.py -v`
Expected: FAIL — `p_sell_bess` is currently in the returned dict.

- [ ] **Step 3: Remove `p_sell_bess` from `_build_lp_data`**

In `src/evopt/controllers/lp_controller.py`, inside `_build_lp_data`, delete this line:

```python
"p_sell_bess": state.get("p_sell_bess", state["p_sell"]),
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_lp_data.py -v`
Expected: 1 passed

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_lp_data.py src/evopt/controllers/lp_controller.py
git commit -m "fix: remove p_sell_bess from LPController data dict"
```

---

### Task 4: Remove `p_sell_bess` from `chargax_wrapper.py`

**Files:**
- Create: `tests/test_wrapper_state.py`
- Modify: `src/evopt/env/chargax_wrapper.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_wrapper_state.py`:

```python
import types
import jax.numpy as jnp


def test_wrapper_extract_state_has_no_p_sell_bess():
    """extract_state should not include p_sell_bess in the returned state dict."""
    from evopt.env.chargax_wrapper import ChargaxWrapper

    wrapper = ChargaxWrapper(n_ports=1, v=400.0, i_max=32.0, p_max_kw=6.0)

    evse = types.SimpleNamespace(
        charger_is_car_connected=jnp.array([False]),
        car_battery_now_kw=jnp.array([0.0]),
        car_battery_capacity_kw=jnp.array([20.0]),
        car_desired_battery_percentage=jnp.array([0.8]),
        car_time_till_leave=jnp.array([60.0]),
    )
    obs = {
        "evses": evse,
        "future_buy_prices":  [0.20] * 24,
        "future_sell_prices": [0.15] * 24,
    }
    chargax_state = types.SimpleNamespace(timestep=0)

    state = wrapper.extract_state(obs, chargax_state)
    assert "p_sell_bess" not in state
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/test_wrapper_state.py -v`
Expected: FAIL — state currently contains `"p_sell_bess"`.

- [ ] **Step 3: Remove `p_sell_bess` from `extract_state`**

In `src/evopt/env/chargax_wrapper.py`, in `extract_state`, make three deletions:

**1. Remove the comment line:**
```python
# p_sell_bess = grid sell price for BESS arbitrage (always market price)
```

**2. Remove the variable declaration:**
```python
p_sell_bess: dict[int, float] = {}
```

**3. Remove the per-step assignment inside the `for h, (bp, sp) in ...` loop:**
```python
p_sell_bess[step] = sp  # BESS always uses market sell price
```

**4. Remove the key from the returned `state` dict:**
```python
"p_sell_bess":   p_sell_bess,
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_wrapper_state.py -v`
Expected: 1 passed

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_wrapper_state.py src/evopt/env/chargax_wrapper.py
git commit -m "fix: remove p_sell_bess from ChargaxWrapper state"
```

---

### Task 5: Fix `runner.py` BESS cost tracking

**Files:**
- Create: `tests/test_runner_bess.py`
- Modify: `src/evopt/benchmarking/runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_runner_bess.py`:

```python
import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock


def _make_mock_runner(bess_net_amps: float, p_buy: float = 0.30,
                      delta_t: float = 5 / 60.0):
    """Build a mocked BenchmarkRunner for a single-step episode with no EVs."""
    from evopt.benchmarking.runner import BenchmarkRunner

    J = 2
    t = 0

    wrapper = MagicMock()
    wrapper.v_bess = 400.0

    clean_state = {
        "t": t,
        "delta_t": delta_t,
        "J": J,
        "P_max": 20_000.0,
        "V": {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0, 2: 32.0},
        "p_buy":  {t: p_buy},
        "p_sell": {t: 0.50},
        "present_cars": {},
        "assignments": {},
        "departed_socs": [],
    }
    wrapper.extract_state.return_value = clean_state
    wrapper.to_chargax_actions.return_value = {
        "evses": [jnp.array([5, 5], dtype=jnp.int32)],
        "batteries": [],
    }

    controller = MagicMock()
    controller.compute_action.return_value = {J + 1: bess_net_amps}

    mock_state = MagicMock()
    mock_state.profit = 0.0
    mock_state.served_customers = 0
    mock_state.rejected_customers = 0
    mock_state.datetime = "2024-01-01"

    mock_timestep = MagicMock()
    mock_timestep.reward = 0.0
    mock_timestep.observation = {}
    mock_timestep.terminated = jnp.array(True)
    mock_timestep.truncated = jnp.array(False)

    env = MagicMock()
    env.reset_env.return_value = ({}, mock_state)
    env.step_env.return_value = (mock_timestep, mock_state)

    return BenchmarkRunner(env, wrapper), controller


def test_runner_bess_charging_adds_to_step_cost():
    """BESS charging (bess_net < 0) should add energy × p_buy to step_cost."""
    # 25 A charging at 400 V for 5 min: energy = 25 * 400 * (5/60) / 1000 = 0.0833 kWh
    # step_cost = 0.0833 × 0.30 = 0.025 €
    delta_t = 5 / 60.0
    p_buy = 0.30
    # LP convention: bess_net = I_dis − I_ch → charging 25 A gives bess_net = −25
    runner, ctrl = _make_mock_runner(bess_net_amps=-25.0, p_buy=p_buy, delta_t=delta_t)
    result = runner.run_episode(ctrl, seed=0)

    expected = 25.0 * 400.0 * delta_t / 1000.0 * p_buy
    assert result.step_log[0].step_cost == pytest.approx(expected, rel=1e-3)


def test_runner_bess_discharging_reduces_step_cost():
    """BESS discharging (bess_net > 0) should subtract energy × p_buy from step_cost."""
    delta_t = 5 / 60.0
    p_buy = 0.30
    runner, ctrl = _make_mock_runner(bess_net_amps=25.0, p_buy=p_buy, delta_t=delta_t)
    result = runner.run_episode(ctrl, seed=0)

    expected = -25.0 * 400.0 * delta_t / 1000.0 * p_buy   # negative = savings
    assert result.step_log[0].step_cost == pytest.approx(expected, rel=1e-3)


def test_runner_no_bess_step_cost_unaffected():
    """When wrapper.v_bess is None, BESS actions should not change step_cost."""
    from evopt.benchmarking.runner import BenchmarkRunner

    J = 2
    t = 0
    delta_t = 5 / 60.0

    wrapper = MagicMock()
    wrapper.v_bess = None

    clean_state = {
        "t": t,
        "delta_t": delta_t,
        "J": J,
        "P_max": 20_000.0,
        "V": {1: 400.0, 2: 400.0},
        "I_max": {1: 32.0, 2: 32.0},
        "p_buy":  {t: 0.30},
        "p_sell": {t: 0.50},
        "present_cars": {},
        "assignments": {},
        "departed_socs": [],
    }
    wrapper.extract_state.return_value = clean_state
    wrapper.to_chargax_actions.return_value = {
        "evses": [jnp.array([5, 5], dtype=jnp.int32)],
        "batteries": [],
    }

    controller = MagicMock()
    controller.compute_action.return_value = {J + 1: -25.0}

    mock_state = MagicMock()
    mock_state.profit = 0.0
    mock_state.served_customers = 0
    mock_state.rejected_customers = 0
    mock_state.datetime = "2024-01-01"

    mock_timestep = MagicMock()
    mock_timestep.reward = 0.0
    mock_timestep.observation = {}
    mock_timestep.terminated = jnp.array(True)
    mock_timestep.truncated = jnp.array(False)

    env = MagicMock()
    env.reset_env.return_value = ({}, mock_state)
    env.step_env.return_value = (mock_timestep, mock_state)

    runner = BenchmarkRunner(env, wrapper)
    result = runner.run_episode(controller, seed=0)

    assert result.step_log[0].step_cost == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run tests to confirm which fail**

Run: `pytest tests/test_runner_bess.py -v`
Expected:
- `test_runner_bess_charging_adds_to_step_cost` — FAIL (`step_cost=0`, expected≈0.025)
- `test_runner_bess_discharging_reduces_step_cost` — FAIL (`step_cost=0`, expected≈−0.025)
- `test_runner_no_bess_step_cost_unaffected` — PASS (no BESS, no EVs → `step_cost=0` already)

- [ ] **Step 3: Add BESS cost block to `runner.py`**

In `src/evopt/benchmarking/runner.py`, inside `run_episode`, immediately after the closing of the `for car_id, port_j in clean_state["assignments"].items():` loop (after the existing `step_cost += energy_kwh * clean_state["p_buy"].get(t, 0.0)` line), add:

```python
            bess_port = clean_state["J"] + 1
            bess_net_amps = actions.get(bess_port, 0.0)
            if self.wrapper.v_bess is not None and bess_net_amps != 0.0:
                bess_net_kwh = -bess_net_amps * self.wrapper.v_bess * delta_t / 1000.0
                step_cost += bess_net_kwh * clean_state["p_buy"].get(t, 0.0)
```

Note the indentation: this block is at the same level as the `for car_id ...` loop (inside `while not done:`), not nested inside the loop.

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_runner_bess.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/test_runner_bess.py src/evopt/benchmarking/runner.py
git commit -m "fix: include BESS net energy in runner step cost tracking"
```
