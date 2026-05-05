# BESS Integration Design

**Date:** 2026-05-05
**Status:** Approved, pending implementation
**Branch:** chargax-integration

## Context

The Chargax benchmark harness is operational with EV charging controllers
(MILP, equal-share, max-charge, random). This spec adds an on-site stationary
battery (BESS) for energy arbitrage: the LP controller jointly optimises EV
charging schedules and BESS charge/discharge decisions in a single solve.

The LP model (`optimization/model.py`) already has full BESS support —
variables `I_bess_ch`, `I_bess_dis`, `SoCB`, port `j_bess = J + 1`. The gap
is that `LPController`, `ChargaxWrapper`, and the station config never wired
it up.

### Decisions made during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Battery type | On-site stationary BESS | Not V2G; `allow_discharging=False` unchanged |
| BESS role | Energy arbitrage | Charge off-peak, discharge on-peak |
| LP formulation | Current-domain (A) with nominal V_bess = 400 V | Consistent with EV port formulation throughout |
| BESS SoC source | Chargax obs every step | No controller-side SoC integration; ground truth from simulator |
| Re-solve cadence | Every `horizon_steps` (fix existing bug) | Current code re-solves every step |
| Background load L | Zero | Simplification; extend later |
| SoC-dependent ratios | `r_bess_ch = r_bess_dis = r_car = 1.0` | Simplification; extend later |

---

## Default BESS Parameters

Sized for the reference station: 3 EV ports × 32 A × 400 V = 38.4 kW peak
EV demand against a 20 kW grid cap.

| Parameter | Value | Notes |
|---|---|---|
| Capacity | 30 kWh | ~3 h at full power; stores one off-peak window |
| Max charge/discharge power | 10 kW | 50% of grid cap |
| Efficiency | 0.95 | ~90% round-trip |
| Initial SoC | 15 kWh | 50% |
| Min SoC | 3 kWh | 10% — battery health floor |
| Nominal DC voltage (LP) | 400 V | Gives I_high = I_low = 25 A |

---

## Unit Conventions

The LP uses current (A) as its decision variable throughout. The BESS is
modelled as port `j_bess = J + 1` with a fixed nominal voltage `V_bess`.

| Quantity | LP unit | Conversion |
|---|---|---|
| EV charging current | A (`I_ev[j,t] ≥ 0`) | → Chargax level via `discretize_amps` |
| BESS charging | A (`I_bess_ch[t] ≥ 0`) | × V_bess / 1000 → kW |
| BESS discharging | A (`I_bess_dis[t] ≥ 0`) | × V_bess / 1000 → kW |
| BESS net (grid view) | A | `I_dis − I_ch`, positive = discharging |
| BESS SoC | kWh | `obs["batteries"][0].battery_now` direct |
| BESS capacity | kWh | `obs["batteries"][0].capacity_kw` (Chargax field name is misleading — it stores kWh) |

**Sign convention note:** Chargax likely uses positive battery action level =
charging (battery gaining energy), negative = discharging. This is the opposite
of the LP's `(I_dis − I_ch)` sign (positive = discharging). The
`to_chargax_actions` conversion must negate. Verify against Chargax source
during implementation.

---

## Architecture

```
Chargax obs["batteries"][0]
    .battery_now  → state["socb_now"]  → LPController._solve(socb_now=...)
    .capacity_kw  → wrapper holds at construction (static field)

LPController._build_lp_data(state):
    assignments, dep, s_cap, s_min, P_car_max, r_car  ← per-car
    SoCB_init  = state["socb_now"]
    SoCB_min, SoCB_max, I_high, I_low                 ← from constructor
    r_bess_ch = r_bess_dis = {t: 1.0}                 ← no SoC derating
    L = {t: 0.0}                                       ← no background load
    V = {1..J: V_ev, J+1: V_bess}

build_rolling_model(data, t, horizon, assignments, soc_now, socb_now)
    → solves for m.I_ev[j,t], m.I_bess_ch[t], m.I_bess_dis[t]

Plan: {step: {port_1..J: ev_amps, port_J+1: bess_net_amps}}

ChargaxWrapper.to_chargax_actions({..., J+1: bess_net_amps}):
    evses[j]     = discretize_amps(amps, I_max[j], num_levels)
    batteries[0] = discretize_bess(bess_net_amps, V_bess, P_bess_max_kw, num_levels)
```

---

## Component Interfaces

### `station_configs.py` — new function

```python
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
```

Adds a `StationBattery(capacity_kw=batt_capacity_kwh, max_kw_throughput=batt_max_kw, efficiency=batt_efficiency)` to the `connections` list alongside the `EVSE`.

### `ChargaxWrapper` — updated constructor

```python
class ChargaxWrapper:
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
        p_bess_max_kw: float | None = None,  # for action discretization
    ) -> None:
```

When `v_bess is None`, BESS fields are omitted from the state dict and
`to_chargax_actions` always sends `batteries=[0]` (existing behaviour).

### `ChargaxWrapper.extract_state` — additions

When BESS is configured:
```python
batt = obs["batteries"][0]
state["socb_now"]  = float(batt.battery_now)
state["socb_min"]  = self.socb_min
state["socb_max"]  = self.socb_max           # from constructor
state["I_high"]    = self.I_high
state["I_low"]     = self.I_low
state["V_bess"]    = self.v_bess
```

### `ChargaxWrapper.to_chargax_actions` — battery slot

```python
# port J+1 is BESS; bess_net_amps > 0 means discharging in LP convention
bess_net_amps = actions.get(self.J + 1, 0.0)
bess_power_kw = bess_net_amps * self.v_bess / 1000.0  # positive = discharge
# Chargax convention: positive = charge → negate (verify during impl)
level = round(-bess_power_kw / self.p_bess_max_kw * self.num_discretization_levels)
level = max(-self.num_discretization_levels, min(self.num_discretization_levels, level))
batteries = jnp.array([level], dtype=jnp.int32)
```

### `LPController` — rewritten

```python
class LPController(BaseController):
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
    ) -> None: ...

    def reset(self) -> None:
        self._plan = None   # {step: {port_j: amps}}

    def compute_action(self, state: dict) -> dict[int, float]:
        t = state["t"]
        if not state["present_cars"]:
            return {}
        if self._plan is None or t % self.horizon_steps == 0:
            self._plan = self._solve(state)
        return self._plan.get(t, {})
```

**`_build_lp_data(state)` — required keys**

```python
J    = state["J"]          # number of EV ports; BESS is port J+1
cars = list(state["present_cars"].keys())
# window = range(t_start, min(t_start + horizon_steps - 1, T_max) + 1)
{
    "J":        J,
    "T":        287,                  # 24 h × 12 steps/h − 1
    "delta_t":  state["delta_t"],
    "P_max":    state["P_max"],
    "V":        {**state["V"], J+1: self.v_bess},
    "I_max":    state["I_max"],       # EV ports only
    "I_high":   self.I_high,
    "I_low":    self.I_low,
    "p_buy":    state["p_buy"],
    "p_sell":   state["p_sell"],
    "L":        {t: 0.0 for t in window},
    "assignments": state["assignments"],
    "dep":      {cid: c["t_max"]  for cid, c in state["present_cars"].items()},
    "s_cap":    {cid: c["s_cap"]  for cid, c in state["present_cars"].items()},
    "s_min":    {cid: 0.0         for cid in cars},
    "P_car_max":{cid: state["V"][assignments[cid]] * state["I_max"][assignments[cid]]
                 for cid in cars},
    "r_car":    {(cid, t): 1.0 for cid in cars for t in window},
    "SoCB_init": state["socb_now"],
    "SoCB_min":  self.socb_min,
    "SoCB_max":  self.socb_max,
    "r_bess_ch": {t: 1.0 for t in window},
    "r_bess_dis":{t: 1.0 for t in window},
}
```

**Plan extraction after solve**

```python
for step in range(t, t_end + 1):
    actions = {}
    for j in m.J_ev:
        actions[int(j)] = float(value(m.I_ev[j, step]))
    bess_net = float(value(m.I_bess_dis[step])) - float(value(m.I_bess_ch[step]))
    actions[J + 1] = bess_net   # positive = discharging
    plan[step] = actions
```

---

## Files Changed

| File | Change |
|---|---|
| `experiments/station_configs.py` | Add `build_station_with_battery` |
| `env/chargax_wrapper.py` | Optional BESS params in constructor; BESS obs extraction; battery action slot |
| `controllers/lp_controller.py` | Full rewrite: correct LP interface, BESS data dict, re-solve cadence fix |
| `experiments/run_chargax_benchmark.py` | Switch to `build_station_with_battery`, BESS-aware wrapper |
| `experiments/run_horizon_comparison.py` | Same |
| `tests/test_chargax_wrapper.py` | Add BESS extraction and action tests |
| `tests/test_lp_controller.py` | Rewrite for new LP interface + BESS state |

---

## Risks and Pitfalls

1. **Chargax battery action sign convention** — verify whether positive
   `batteries[0]` means charging or discharging before wiring up
   `to_chargax_actions`. Check `StationBattery.throughput_now_kw` sign in the
   Chargax source.

2. **`capacity_kw` naming** — Chargax names the battery capacity field
   `capacity_kw` but its value is in kWh. Use it directly as `SoCB_max` in kWh.

3. **`dep[i]` quality** — `t_max` from `car_time_till_leave` is an estimate.
   If the estimate is wrong (e.g., car leaves earlier than predicted), the LP
   may plan EV charging past the actual departure. The rolling re-solve every
   `horizon_steps` mitigates this by updating `dep` from fresh observations.

4. **Single-battery assumption** — `obs["batteries"][0]` assumes exactly one
   BESS. If the station has multiple batteries, the wrapper silently ignores
   them. Add an assertion in the constructor.

5. **`r_car[i, t]` window scope** — the data dict must include `r_car` for
   every `(car_id, t)` in `[t_start, t_end]`. Forgetting steps outside the
   immediate window causes a `KeyError` in constraint C9.
