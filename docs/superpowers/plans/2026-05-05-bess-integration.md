# BESS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire an on-site stationary BESS into the Chargax rolling-horizon LP benchmark so the LP jointly optimises EV charging and battery arbitrage in a single solve.

**Architecture:** `ChargaxWrapper` extracts BESS SoC from observations and converts LP discharge/charge currents into Chargax battery levels. `LPController` is fully rewritten to use the existing LP model (`build_rolling_model`) with the correct data dict, BESS state, and a fixed re-solve cadence. Two experiment scripts are updated to use a battery-equipped station.

**Tech Stack:** Python, Pyomo (LP via HiGHS), Chargax JAX simulator, pytest.

All commands run from `.worktrees/chargax-integration/` unless noted.

---

## File Map

| File | Action |
|---|---|
| `src/evopt/experiments/station_configs.py` | Add `build_station_with_battery` |
| `src/evopt/env/chargax_wrapper.py` | Optional BESS params; BESS obs extraction; battery action slot |
| `src/evopt/controllers/lp_controller.py` | Full rewrite: correct LP interface, BESS data dict, cadence fix |
| `src/evopt/experiments/run_chargax_benchmark.py` | Switch to battery station and BESS wrapper |
| `src/evopt/experiments/run_horizon_comparison.py` | Same |
| `src/evopt/tests/test_chargax_wrapper.py` | Add BESS extraction and action tests |
| `src/evopt/tests/test_lp_controller.py` | Rewrite for new LP interface + BESS state |

---

## Task 1: `build_station_with_battery`

**Files:**
- Modify: `src/evopt/experiments/station_configs.py`
- Test: `src/evopt/tests/test_station_configs.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/evopt/tests/test_station_configs.py`:

```python
from chargax import EVSE, ChargingStation
from chargax.station import StationBattery

from evopt.experiments.station_configs import build_station_with_battery


def test_returns_charging_station():
    st = build_station_with_battery(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    assert isinstance(st, ChargingStation)


def test_has_one_evse_and_one_battery():
    st = build_station_with_battery(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    evses = [c for c in st.connections if isinstance(c, EVSE)]
    batts = [c for c in st.connections if isinstance(c, StationBattery)]
    assert len(evses) == 1
    assert len(batts) == 1


def test_evse_has_correct_charger_count():
    st = build_station_with_battery(n_ports=2, v=400.0, i_max=32.0, p_max_kw=20.0)
    evse = next(c for c in st.connections if isinstance(c, EVSE))
    assert evse.num_chargers == 2


def test_battery_defaults():
    st = build_station_with_battery(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    from chargax.station import StationBattery
    batt = next(c for c in st.connections if isinstance(c, StationBattery))
    assert batt.capacity_kw == 30.0
    assert batt.max_kw_throughput == 10.0
    assert batt.efficiency == 0.95


def test_battery_custom_params():
    st = build_station_with_battery(
        n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0,
        batt_capacity_kwh=50.0, batt_max_kw=15.0, batt_efficiency=0.9,
    )
    from chargax.station import StationBattery
    batt = next(c for c in st.connections if isinstance(c, StationBattery))
    assert batt.capacity_kw == 50.0
    assert batt.max_kw_throughput == 15.0
    assert batt.efficiency == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest src/evopt/tests/test_station_configs.py -v
```

Expected: `ImportError` or `AttributeError` — `build_station_with_battery` does not exist yet.

- [ ] **Step 3: Implement `build_station_with_battery`**

Edit `src/evopt/experiments/station_configs.py`:

```python
from chargax import EVSE, ChargingStation
from chargax.station import StationBattery


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
        efficiency=1.0,
        connections=[evse],
    )


def build_station_with_battery(
    n_ports: int,
    v: float,
    i_max: float,
    p_max_kw: float,
    batt_capacity_kwh: float = 30.0,
    batt_max_kw: float = 10.0,
    batt_efficiency: float = 0.95,
) -> ChargingStation:
    """Station with n_ports EV chargers and one on-site BESS."""
    evse = EVSE(
        num_chargers=n_ports,
        voltage=v,
        max_current=i_max,
        efficiency=1.0,
    )
    batt = StationBattery(
        capacity_kw=batt_capacity_kwh,
        max_kw_throughput=batt_max_kw,
        efficiency=batt_efficiency,
    )
    return ChargingStation(
        max_kw_throughput=p_max_kw,
        efficiency=1.0,
        connections=[evse, batt],
    )
```

> **Note:** `StationBattery` may live at a different import path. If `from chargax.station import StationBattery` fails, try `from chargax import StationBattery`. Verify against the Chargax source before committing. The field name `capacity_kw` on `StationBattery` stores kWh despite its name — this is a known Chargax quirk documented in the design spec.

- [ ] **Step 4: Run test to verify it passes**

```
pytest src/evopt/tests/test_station_configs.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evopt/experiments/station_configs.py src/evopt/tests/test_station_configs.py
git commit -m "feat: add build_station_with_battery to station_configs"
```

---

## Task 2: `ChargaxWrapper` — BESS constructor, obs extraction, action conversion

**Files:**
- Modify: `src/evopt/env/chargax_wrapper.py`
- Modify: `src/evopt/tests/test_chargax_wrapper.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/evopt/tests/test_chargax_wrapper.py`:

```python
# ── BESS test helpers ────────────────────────────────────────────────────────

class MockBattery:
    def __init__(self, battery_now: float, capacity_kw: float):
        self.battery_now = battery_now
        self.capacity_kw = capacity_kw   # misleading name; value is in kWh


def _make_obs_with_bess(evse, battery, buy_prices=None, sell_prices=None):
    obs = _make_obs(evse, buy_prices, sell_prices)
    obs["batteries"] = [battery]
    return obs


def _make_bess_wrapper(n_ports: int = 2) -> ChargaxWrapper:
    return ChargaxWrapper(
        n_ports=n_ports,
        v=400.0,
        i_max=32.0,
        p_max_kw=20.0,
        num_discretization_levels=10,
        minutes_per_step=5,
        v_bess=400.0,
        I_high=25.0,
        I_low=25.0,
        socb_min=3.0,
        socb_max=30.0,
        p_bess_max_kw=10.0,
    )


# ── BESS obs extraction tests ────────────────────────────────────────────────

def test_bess_wrapper_constructor_stores_params():
    w = _make_bess_wrapper()
    assert w.v_bess == 400.0
    assert w.I_high == 25.0
    assert w.I_low == 25.0
    assert w.socb_min == 3.0
    assert w.socb_max == 30.0
    assert w.p_bess_max_kw == 10.0


def test_extract_state_includes_bess_fields():
    w = _make_bess_wrapper(1)
    evse = MockEVSE(1, [False], [0.0], [60.0], [0.0], [0.0])
    batt = MockBattery(battery_now=15.0, capacity_kw=30.0)
    state = w.extract_state(_make_obs_with_bess(evse, batt), MockState(0))
    assert state["socb_now"] == pytest.approx(15.0)
    assert state["socb_min"] == pytest.approx(3.0)
    assert state["socb_max"] == pytest.approx(30.0)
    assert state["I_high"]   == pytest.approx(25.0)
    assert state["I_low"]    == pytest.approx(25.0)
    assert state["V_bess"]   == pytest.approx(400.0)


def test_extract_state_without_bess_has_no_bess_fields():
    w = _make_wrapper(1)   # no BESS params
    evse = MockEVSE(1, [False], [0.0], [60.0], [0.0], [0.0])
    state = w.extract_state(_make_obs(evse), MockState(0))
    assert "socb_now" not in state


# ── BESS action conversion tests ─────────────────────────────────────────────

def test_to_chargax_actions_bess_discharging():
    # LP convention: positive bess_net_amps = discharging
    # bess_net_amps = 25 A at 400 V = 10 kW (full discharge)
    # Chargax: positive = charging → negate → level = -10
    w = _make_bess_wrapper(2)
    actions = {1: 0.0, 2: 0.0, 3: 25.0}   # port 3 = BESS (J+1 for J=2)
    result = w.to_chargax_actions(actions)
    batt_level = int(result["batteries"][0][0])
    assert batt_level == -10   # full discharge


def test_to_chargax_actions_bess_charging():
    # bess_net_amps = -25 A (charging in LP terms; I_dis=0, I_ch=25)
    # bess_power_kw = -25 * 400 / 1000 = -10 kW (negative = charge)
    # level = round(-(-10) / 10 * 10) = +10
    w = _make_bess_wrapper(2)
    actions = {1: 0.0, 2: 0.0, 3: -25.0}  # negative → charging in LP terms
    result = w.to_chargax_actions(actions)
    batt_level = int(result["batteries"][0][0])
    assert batt_level == 10   # full charge


def test_to_chargax_actions_bess_idle():
    w = _make_bess_wrapper(2)
    actions = {1: 32.0, 2: 16.0}   # no BESS key
    result = w.to_chargax_actions(actions)
    batt_level = int(result["batteries"][0][0])
    assert batt_level == 0


def test_to_chargax_actions_bess_level_clamped():
    # Very large discharge beyond p_bess_max_kw should clamp to -num_levels
    w = _make_bess_wrapper(2)
    actions = {3: 1000.0}   # unrealistically large
    result = w.to_chargax_actions(actions)
    batt_level = int(result["batteries"][0][0])
    assert batt_level == -10   # clamped


def test_to_chargax_actions_no_bess_returns_empty_batteries():
    w = _make_wrapper(2)   # no BESS
    result = w.to_chargax_actions({1: 32.0, 2: 16.0})
    assert result["batteries"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_chargax_wrapper.py -v -k "bess"
```

Expected: multiple failures — BESS params not accepted by constructor yet.

- [ ] **Step 3: Implement BESS additions in `ChargaxWrapper`**

Replace `src/evopt/env/chargax_wrapper.py` entirely:

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
        The LP uses 1-indexed port IDs (j = 1, 2, ..., J).
        BESS occupies port J+1 in the LP; it has no Chargax charger slot.
    """

    def __init__(
        self,
        n_ports: int,
        v: float,
        i_max: float,
        p_max_kw: float,
        num_discretization_levels: int = 10,
        minutes_per_step: int = 5,
        # BESS params (optional — if None, BESS is ignored)
        v_bess: float | None = None,
        I_high: float | None = None,   # max BESS charging current (A)
        I_low: float | None = None,    # max BESS discharging current (A)
        socb_min: float = 0.0,
        socb_max: float | None = None,
        p_bess_max_kw: float | None = None,
    ) -> None:
        self.J                         = n_ports
        self.V                         = {j + 1: v     for j in range(n_ports)}
        self.I_max                     = {j + 1: i_max for j in range(n_ports)}
        self.P_max_w                   = p_max_kw * 1000.0
        self.num_discretization_levels = num_discretization_levels
        self.minutes_per_step          = minutes_per_step

        self.v_bess       = v_bess
        self.I_high       = I_high
        self.I_low        = I_low
        self.socb_min     = socb_min
        self.socb_max     = socb_max
        self.p_bess_max_kw = p_bess_max_kw

        self._charger_to_car: dict[int, int] = {}
        self._prev_connected: set[int]        = set()
        self._prev_soc: dict[int, float]      = {}
        self._next_car_id: int                = 0

    def reset(self) -> None:
        self._charger_to_car = {}
        self._prev_connected = set()
        self._prev_soc       = {}
        self._next_car_id    = 0

    def extract_state(self, obs: dict, chargax_state) -> dict:
        evse_raw = obs["evses"]
        evse = evse_raw[0] if isinstance(evse_raw, list) else evse_raw
        t    = int(chargax_state.timestep)

        now_connected = {
            j for j in range(self.J)
            if bool(evse.charger_is_car_connected[j])
        }
        departures   = self._prev_connected - now_connected
        new_arrivals = now_connected - self._prev_connected

        departed_socs: list[float] = []
        for j in sorted(departures):
            departed_socs.append(self._prev_soc.get(j, float(evse.car_battery_now_kw[j])))
            del self._charger_to_car[j]
            self._prev_soc.pop(j, None)

        for j in sorted(new_arrivals):
            self._charger_to_car[j] = self._next_car_id
            self._next_car_id += 1

        self._prev_connected = now_connected

        present_cars: dict = {}
        assignments:  dict = {}
        for j, car_id in self._charger_to_car.items():
            soc_now       = float(evse.car_battery_now_kw[j])
            s_cap         = float(evse.car_battery_capacity_kw[j])
            desired_pct   = float(evse.car_desired_battery_percentage[j])
            s_target      = desired_pct * s_cap
            time_left_min = float(evse.car_time_till_leave[j])
            t_max         = t + max(1, round(time_left_min / self.minutes_per_step))

            present_cars[car_id] = {
                "soc_now":  soc_now,
                "s_target": s_target,
                "s_cap":    s_cap,
                "t_max":    t_max,
            }
            assignments[car_id] = j + 1
            self._prev_soc[j]   = soc_now

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

        state: dict = {
            "t":             t,
            "delta_t":       self.minutes_per_step / 60.0,
            "J":             self.J,
            "P_max":         self.P_max_w,
            "V":             self.V,
            "I_max":         self.I_max,
            "p_buy":         p_buy,
            "p_sell":        p_sell,
            "present_cars":  present_cars,
            "assignments":   assignments,
            "departed_socs": departed_socs,
        }

        if self.v_bess is not None:
            batt = obs["batteries"][0]
            state["socb_now"] = float(batt.battery_now)
            state["socb_min"] = self.socb_min
            state["socb_max"] = self.socb_max
            state["I_high"]   = self.I_high
            state["I_low"]    = self.I_low
            state["V_bess"]   = self.v_bess

        return state

    def to_chargax_actions(self, actions: dict[int, float]) -> dict:
        """Convert {port_j (1-indexed): amps} to Chargax MultiDiscrete action dict.

        Port J+1 is the BESS. LP convention: positive bess_net_amps = discharging.
        Chargax convention: positive battery level = charging. Signs are negated.
        """
        levels = []
        for j in range(self.J):
            port_j = j + 1
            amps   = actions.get(port_j, 0.0)
            levels.append(discretize_amps(amps, self.I_max[port_j],
                                          self.num_discretization_levels))

        if self.v_bess is not None:
            bess_net_amps  = actions.get(self.J + 1, 0.0)
            bess_power_kw  = bess_net_amps * self.v_bess / 1000.0   # positive = discharge
            level = round(-bess_power_kw / self.p_bess_max_kw * self.num_discretization_levels)
            level = max(-self.num_discretization_levels,
                        min(self.num_discretization_levels, level))
            batteries = [jnp.array([level], dtype=jnp.int32)]
        else:
            batteries = []

        return {
            "evses":     [jnp.array(levels, dtype=jnp.int32)],
            "batteries": batteries,
        }
```

- [ ] **Step 4: Run all wrapper tests**

```
pytest src/evopt/tests/test_chargax_wrapper.py -v
```

Expected: all tests PASS (both original and new BESS tests).

- [ ] **Step 5: Commit**

```bash
git add src/evopt/env/chargax_wrapper.py src/evopt/tests/test_chargax_wrapper.py
git commit -m "feat: add optional BESS support to ChargaxWrapper"
```

---

## Task 3: `LPController` — full rewrite

**Files:**
- Modify: `src/evopt/controllers/lp_controller.py`
- Modify: `src/evopt/tests/test_lp_controller.py`

- [ ] **Step 1: Write the failing tests**

Replace `src/evopt/tests/test_lp_controller.py` entirely:

```python
import pytest
from evopt.controllers.lp_controller import LPController


def _make_state(t: int, n_cars: int = 1, p_max_w: float = 10_000.0,
                socb_now: float = 15.0) -> dict:
    """Minimal Chargax-format state with n_cars and BESS."""
    present_cars = {
        i + 1: {"soc_now": 5.0, "s_target": 10.0, "s_cap": 60.0, "t_max": t + 24}
        for i in range(n_cars)
    }
    assignments = {i + 1: i + 1 for i in range(n_cars)}
    J = max(n_cars, 1)
    p_buy  = {s: 0.20 for s in range(t, t + 100)}
    p_sell = {s: 0.40 for s in range(t, t + 100)}
    return {
        "t":             t,
        "delta_t":       5 / 60,
        "J":             J,
        "P_max":         p_max_w,
        "V":             {j: 400.0 for j in range(1, J + 1)},
        "I_max":         {j: 32.0  for j in range(1, J + 1)},
        "p_buy":         p_buy,
        "p_sell":        p_sell,
        "present_cars":  present_cars,
        "assignments":   assignments,
        "departed_socs": [],
        "socb_now":      socb_now,
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


def test_bess_action_in_plan():
    """BESS port J+1 must appear in the plan."""
    ctrl = LPController(horizon_steps=6, solver="highs")
    state = _make_state(0, n_cars=1)
    ctrl.compute_action(state)
    J = state["J"]
    t = state["t"]
    assert (J + 1) in ctrl._plan[t]


def test_plan_cached_within_horizon():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    plan_after_t0 = ctrl._plan

    state1 = _make_state(1)
    ctrl.compute_action(state1)
    assert ctrl._plan is plan_after_t0   # no re-solve between horizon boundaries


def test_resolves_at_horizon_boundary():
    ctrl = LPController(horizon_steps=12, solver="highs")
    state0 = _make_state(0)
    ctrl.compute_action(state0)
    plan_after_t0 = ctrl._plan

    state12 = _make_state(12)
    ctrl.compute_action(state12)
    assert ctrl._plan is not plan_after_t0   # new solve at t=12


def test_resolves_when_plan_is_none():
    ctrl = LPController(horizon_steps=12, solver="highs")
    ctrl._plan = None
    state = _make_state(5)
    ctrl.compute_action(state)
    assert ctrl._plan is not None


def test_build_lp_data_has_required_keys():
    ctrl = LPController(horizon_steps=6, solver="highs")
    state = _make_state(0, n_cars=1)
    data = ctrl._build_lp_data(state)
    required = {
        "J", "T", "delta_t", "P_max", "V", "I_max",
        "I_high", "I_low", "p_buy", "p_sell", "L",
        "assignments", "dep", "s_cap", "s_min", "P_car_max",
        "r_car", "SoCB_init", "SoCB_min", "SoCB_max",
        "r_bess_ch", "r_bess_dis",
    }
    assert required <= data.keys()


def test_build_lp_data_bess_port_in_V():
    ctrl = LPController(horizon_steps=6, solver="highs")
    state = _make_state(0, n_cars=1)
    data = ctrl._build_lp_data(state)
    J = state["J"]
    assert (J + 1) in data["V"]
    assert data["V"][J + 1] == pytest.approx(ctrl.v_bess)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest src/evopt/tests/test_lp_controller.py -v
```

Expected: multiple failures — controller uses wrong variable names and missing BESS params.

- [ ] **Step 3: Implement the rewritten `LPController`**

Replace `src/evopt/controllers/lp_controller.py` entirely:

```python
from __future__ import annotations

from pyomo.environ import value

from evopt.controllers.base_controller import BaseController
from evopt.optimization.model import build_rolling_model
from evopt.optimization.solver import solve


class LPController(BaseController):
    """
    Stateful rolling MPC controller.

    Re-solves the LP at t=0 and every `horizon_steps` thereafter, caching
    the plan for intermediate steps. Jointly optimises EV charging and BESS
    arbitrage in a single LP solve.
    """

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
    ) -> None:
        self.horizon_steps    = horizon_steps
        self.solver           = solver
        self.v_bess           = v_bess
        self.I_high           = I_high
        self.I_low            = I_low
        self.socb_min         = socb_min
        self.socb_max         = socb_max
        self.minutes_per_step = minutes_per_step

        self._plan: dict[int, dict[int, float]] | None = None

    def reset(self) -> None:
        self._plan = None

    def compute_action(self, state: dict) -> dict[int, float]:
        t = state["t"]

        if not state["present_cars"]:
            return {}

        if self._plan is None or t % self.horizon_steps == 0:
            self._plan = self._solve(state)

        return self._plan.get(t, {})

    def _build_lp_data(self, state: dict) -> dict:
        T_max = 24 * 60 // self.minutes_per_step - 1   # 287 for 5-min steps

        t          = state["t"]
        J          = state["J"]
        cars       = list(state["present_cars"].keys())
        assignments = state["assignments"]

        t_end  = min(t + self.horizon_steps - 1, T_max)
        window = range(t, t_end + 1)

        return {
            "J":        J,
            "T":        T_max,
            "delta_t":  state["delta_t"],
            "P_max":    state["P_max"],
            "V":        {**state["V"], J + 1: self.v_bess},
            "I_max":    state["I_max"],
            "I_high":   self.I_high,
            "I_low":    self.I_low,
            "p_buy":    state["p_buy"],
            "p_sell":   state["p_sell"],
            "L":        {step: 0.0 for step in window},
            "assignments": assignments,
            "dep":      {cid: c["t_max"] for cid, c in state["present_cars"].items()},
            "s_cap":    {cid: c["s_cap"] for cid, c in state["present_cars"].items()},
            "s_min":    {cid: 0.0 for cid in cars},
            "P_car_max": {
                cid: state["V"][assignments[cid]] * state["I_max"][assignments[cid]]
                for cid in cars
            },
            "r_car":     {(cid, step): 1.0 for cid in cars for step in window},
            "SoCB_init": state["socb_now"],
            "SoCB_min":  self.socb_min,
            "SoCB_max":  self.socb_max,
            "r_bess_ch": {step: 1.0 for step in window},
            "r_bess_dis":{step: 1.0 for step in window},
        }

    def _solve(self, state: dict) -> dict[int, dict[int, float]]:
        t           = state["t"]
        J           = state["J"]
        assignments = state["assignments"]
        soc_now     = {cid: c["soc_now"] for cid, c in state["present_cars"].items()}
        socb_now    = state["socb_now"]
        data        = self._build_lp_data(state)

        m = build_rolling_model(data, t, self.horizon_steps, assignments, soc_now, socb_now)
        if m is None:
            return {}

        solve(m, solver=self.solver)

        T_max = data["T"]
        t_end = min(t + self.horizon_steps - 1, T_max)

        plan: dict[int, dict[int, float]] = {}
        for step in range(t, t_end + 1):
            actions: dict[int, float] = {}
            for j in m.J_ev:
                actions[int(j)] = float(value(m.I_ev[j, step]))
            bess_net = float(value(m.I_bess_dis[step])) - float(value(m.I_bess_ch[step]))
            actions[J + 1] = bess_net   # positive = discharging (LP convention)
            plan[step] = actions

        return plan
```

- [ ] **Step 4: Run all LP controller tests**

```
pytest src/evopt/tests/test_lp_controller.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```
pytest src/evopt/tests/ -v --tb=short
```

Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evopt/controllers/lp_controller.py src/evopt/tests/test_lp_controller.py
git commit -m "feat: rewrite LPController for LP model interface with BESS and fixed re-solve cadence"
```

---

## Task 4: Update experiment scripts

**Files:**
- Modify: `src/evopt/experiments/run_chargax_benchmark.py`
- Modify: `src/evopt/experiments/run_horizon_comparison.py`

No new tests — the scripts are entry points. Correctness is verified by the smoke test in Task 5.

- [ ] **Step 1: Update `run_chargax_benchmark.py`**

Replace `src/evopt/experiments/run_chargax_benchmark.py`:

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
from evopt.experiments.station_configs import build_station_with_battery


def main(n_seeds: int = 10, output_dir: Path = Path("results/chargax")) -> None:
    N_PORTS       = 3
    VOLTAGE       = 400.0
    I_MAX         = 32.0
    P_MAX_KW      = 20.0
    V_BESS        = 400.0
    P_BESS_MAX_KW = 10.0
    I_BESS        = P_BESS_MAX_KW * 1000.0 / V_BESS   # 25 A

    station = build_station_with_battery(
        n_ports=N_PORTS, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
        batt_capacity_kwh=30.0, batt_max_kw=P_BESS_MAX_KW, batt_efficiency=0.95,
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
        v_bess=V_BESS,
        I_high=I_BESS,
        I_low=I_BESS,
        socb_min=3.0,
        socb_max=30.0,
        p_bess_max_kw=P_BESS_MAX_KW,
    )

    runner = BenchmarkRunner(env, wrapper)
    runner.run_benchmark(
        controllers={
            "milp_h12":    LPController(
                horizon_steps=12, solver="highs",
                v_bess=V_BESS, I_high=I_BESS, I_low=I_BESS,
                socb_min=3.0, socb_max=30.0,
            ),
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

- [ ] **Step 2: Update `run_horizon_comparison.py`**

Replace `src/evopt/experiments/run_horizon_comparison.py`:

```python
"""
run_horizon_comparison.py — Compare LPController performance across horizon lengths.

Usage:
    python -m evopt.experiments.run_horizon_comparison
    python -m evopt.experiments.run_horizon_comparison --seeds 5 --output results/horizon
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chargax import Chargax

from evopt.benchmarking.runner import BenchmarkRunner
from evopt.benchmarking.storage import build_summary
from evopt.controllers.lp_controller import LPController
from evopt.env.chargax_wrapper import ChargaxWrapper
from evopt.experiments.station_configs import build_station_with_battery


def main(n_seeds: int = 5, output_dir: Path = Path("results/horizon")) -> None:
    N_PORTS       = 3
    VOLTAGE       = 400.0
    I_MAX         = 32.0
    P_MAX_KW      = 20.0
    V_BESS        = 400.0
    P_BESS_MAX_KW = 10.0
    I_BESS        = P_BESS_MAX_KW * 1000.0 / V_BESS   # 25 A

    station = build_station_with_battery(
        n_ports=N_PORTS, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
        batt_capacity_kwh=30.0, batt_max_kw=P_BESS_MAX_KW, batt_efficiency=0.95,
    )
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=True,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=N_PORTS, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
        num_discretization_levels=10,
        minutes_per_step=5,
        v_bess=V_BESS,
        I_high=I_BESS,
        I_low=I_BESS,
        socb_min=3.0,
        socb_max=30.0,
        p_bess_max_kw=P_BESS_MAX_KW,
    )

    horizons = [1, 3, 6, 12]
    controllers = {
        f"milp_h{h}": LPController(
            horizon_steps=h, solver="highs",
            v_bess=V_BESS, I_high=I_BESS, I_low=I_BESS,
            socb_min=3.0, socb_max=30.0,
        )
        for h in horizons
    }

    runner = BenchmarkRunner(env, wrapper)
    runner.run_benchmark(
        controllers=controllers,
        seeds=list(range(n_seeds)),
        output_dir=output_dir,
    )

    df = build_summary(output_dir)
    summary = (
        df.groupby("controller")[["net_profit", "served_customers", "rejected_customers"]]
        .mean()
        .reindex([f"milp_h{h}" for h in horizons])
        .round(2)
    )
    print("\nHorizon Comparison (mean across seeds)\n")
    print(f"{'Horizon':<12} {'Net Profit':>12} {'Served':>8} {'Rejected':>10}")
    print("-" * 44)
    for name, row in summary.iterrows():
        h = name.replace("milp_h", "H=")
        print(f"{h:<12} {row['net_profit']:>12.2f} {row['served_customers']:>8.1f} {row['rejected_customers']:>10.1f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int, default=5)
    parser.add_argument("--output", type=str, default="results/horizon")
    args = parser.parse_args()
    main(n_seeds=args.seeds, output_dir=Path(args.output))
```

- [ ] **Step 3: Commit**

```bash
git add src/evopt/experiments/run_chargax_benchmark.py src/evopt/experiments/run_horizon_comparison.py
git commit -m "feat: wire battery station and BESS wrapper into benchmark scripts"
```

---

## Task 5: Smoke test

**Files:** none new — verification only.

- [ ] **Step 1: Run the full test suite**

```
pytest src/evopt/tests/ -v --tb=short
```

Expected: all tests PASS, no errors.

- [ ] **Step 2: Run a single-seed benchmark to verify end-to-end**

```
python -m evopt.experiments.run_chargax_benchmark --seeds 1 --output results/smoke
```

Expected output: a table with `milp_h12`, `equal_share`, `max_charge`, `random` rows and non-NaN `net_profit`. The LP controller should show a higher net profit than `random`.

If the LP shows `net_profit = 0.0` across all seeds, the most likely causes are:
1. `p_sell <= p_buy` in the price data (no arbitrage incentive) — check price range
2. Grid cap constraint too tight — lower `P_MAX_KW` or raise `I_BESS`
3. Solver infeasibility — add `tee=True` to `solve()` call and check HiGHS output

- [ ] **Step 3: Verify BESS sign convention against Chargax source**

Before declaring the integration complete, confirm that positive `batteries[0]` = charging in the Chargax step function. Check `StationBattery.throughput_now_kw` sign convention in the Chargax source:

```bash
python -c "
from chargax import Chargax
from evopt.experiments.station_configs import build_station_with_battery
st = build_station_with_battery(3, 400.0, 32.0, 20.0)
env = Chargax(station=st, minutes_per_timestep=5,
              allow_discharging=False, renormalize_currents=False)
import jax.numpy as jnp
obs, state = env.reset(seed=0)
# Send full charge command (level=10) and observe battery_now change
action = {'evses': [jnp.zeros(3, dtype=jnp.int32)],
          'batteries': [jnp.array([10], dtype=jnp.int32)]}
obs2, state2, *_ = env.step(action, state)
soc_before = float(obs['batteries'][0].battery_now)
soc_after  = float(obs2['batteries'][0].battery_now)
print(f'level=+10: SoC {soc_before:.2f} → {soc_after:.2f}  (positive = charging?)')
"
```

Expected: SoC after > SoC before (confirming positive level = charging = LP negation is correct).
If SoC decreases instead, flip the sign in `to_chargax_actions`: replace `-bess_power_kw` with `bess_power_kw`.

- [ ] **Step 4: Final commit if sign correction was needed**

```bash
git add src/evopt/env/chargax_wrapper.py
git commit -m "fix: correct BESS action sign convention after empirical verification"
```

---

## Self-Review

**Spec coverage:**
- `build_station_with_battery` → Task 1 ✓
- `ChargaxWrapper` BESS constructor + `extract_state` + `to_chargax_actions` → Task 2 ✓
- `LPController` rewrite with `_build_lp_data`, `_solve`, cadence fix → Task 3 ✓
- `run_chargax_benchmark.py` and `run_horizon_comparison.py` → Task 4 ✓
- Sign convention verification → Task 5 Step 3 ✓
- `capacity_kw` naming pitfall → documented in Task 1 Step 3 note ✓
- Single-battery assumption → enforced implicitly by `obs["batteries"][0]` ✓
- `r_car` window scope → `window = range(t, t_end + 1)` matches `build_rolling_model` ✓

**Placeholder scan:** all steps contain concrete code. No TBDs.

**Type consistency:**
- `socb_now` key used consistently in state dict, `_build_lp_data`, and `_solve`.
- `J + 1` is the BESS port throughout (wrapper, controller, model).
- `bess_net_amps` sign (positive = discharge) is consistent between `_solve` plan extraction and `to_chargax_actions` input.
