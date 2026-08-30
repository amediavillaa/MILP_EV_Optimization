# LP Formulation — EV Charging Park with On-Site BESS

## Overview

The optimisation problem is a **pure linear programme (LP)**. Despite the repository name, no binary or integer variables are present in the current formulation. Port assignments and vehicle occupancy are precomputed externally by the Chargax FCFS scheduler; the LP's sole decision is how much current to deliver at each port and to/from the battery at each time step. The LP is solved as a **rolling-horizon MPC**: at every 5-minute step the controller re-solves over a lookahead window of `H` steps using the latest observed state.

An earlier MILP formulation modelled port assignment (`y`), occupancy (`x`), and departure triggering (`delta`) as binary variables. That formulation has been replaced: those variables are preserved in `variables.py` for reference but are not imported or used anywhere.

---

## Sets and Indices

| Symbol | Description |
|--------|-------------|
| `j ∈ {1, …, J}` | EV charging ports |
| `j = J+1` | BESS port (virtual port index) |
| `i ∈ {1, …, I}` | EVs present in the current window |
| `t ∈ {t_start, …, t_start + H − 1}` | Time steps in the lookahead window |

---

## Parameters

### Station hardware

| Symbol | Unit | Description |
|--------|------|-------------|
| `V_j` | V | Port voltage (same for all EV ports; 400 V for BESS) |
| `I_max_j` | A | Maximum charging current for EV port `j` |
| `I_high` | A | BESS maximum charging current |
| `I_low` | A | BESS maximum discharging current |
| `P_max` | W | Grid connection power cap |
| `δt` | h | Time-step duration (5 min = 1/12 h) |

### BESS

| Symbol | Unit | Description |
|--------|------|-------------|
| `SoCB_init` | kWh | BESS state of charge at start of window |
| `SoCB_min` | kWh | BESS minimum SoC (default 3 kWh) |
| `SoCB_max` | kWh | BESS maximum SoC (default 30 kWh) |
| `η` | — | One-way charge/discharge efficiency (default 0.95; round-trip ≈ 0.90) |
| `r_bess_ch(t)` | — | BESS charging de-rating ratio ∈ [0, 1] |
| `r_bess_dis(t)` | — | BESS discharging de-rating ratio ∈ [0, 1] |

### EVs

| Symbol | Unit | Description |
|--------|------|-------------|
| `s_init_i` | kWh | EV `i` SoC on arrival (or at window start) |
| `s_cap_i` | kWh | EV `i` battery capacity (physical maximum) |
| `s_target_i` | kWh | EV `i` requested departure SoC |
| `s_min_i` | kWh | EV `i` minimum SoC (= 0) |
| `dep_i` | step | Departure time step of EV `i` |
| `P_car_max_i` | W | Maximum charging power of EV `i` |
| `r_car(i, t)` | — | SoC-dependent charging ratio ∈ [0, 1] (= 1 everywhere unless derating active) |
| `z(j, t)` | {0,1} | Occupancy indicator: 1 iff port `j` has a car at step `t` |

### Prices and loads

| Symbol | Unit | Description |
|--------|------|-------------|
| `p_buy(t)` | €/kWh | Electricity buy price at step `t` |
| `p_sell(t)` | €/kWh | EV charging tariff (revenue per kWh delivered) — fixed or dynamic |
| `L(t)` | W | Background load at step `t` (assumed 0 — not exposed by Chargax) |

---

## Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| `I_ev(j, t)` | ℝ≥0 | Charging current at EV port `j` at step `t` (A) |
| `I_bess_ch(t)` | ℝ≥0 | BESS charging current at step `t` (A) |
| `I_bess_dis(t)` | ℝ≥0 | BESS discharging current at step `t` (A) |
| `SoCB(t)` | ℝ≥0 | BESS state of charge at end of step `t` (kWh) |
| `soc_car(i, t)` | ℝ≥0 | EV `i` state of charge at end of step `t` (kWh) |

---

## Objective

Maximise net profit = EV charging revenue − grid electricity cost.

```
maximise  Σ_{j,t} V_j · I_ev(j,t) · z(j,t) · p_eff(t) · δt/1000
        − Σ_t [ (Σ_j V_j · I_ev(j,t)  +  V_{J+1} · I_bess_ch(t)/η  −  V_{J+1} · I_bess_dis(t)·η)
                · p_buy(t) · δt/1000 ]
```

where the effective sell price includes a small urgency tie-breaker:

```
p_eff(t) = p_sell(t) + ε · (t_last − t + 1) / H      ε = 1e-4
```

The urgency term `ε` breaks ties so the LP prefers to charge earlier rather than procrastinating to the end of the window.

**BESS grid accounting:** charging draws `I_bess_ch / η` from the grid (efficiency loss on the way in); discharging saves `I_bess_dis · η` of grid draw (efficiency loss on the way out). BESS discharge is not sold externally — it reduces the station's net grid import.

**Tariff modes:**
- *Fixed:* `p_sell(t) = 0.75 €/kWh` constant.
- *Dynamic cost-plus:* `p_sell(t) = p_buy(t) × markup` (e.g. ×1.5).

---

## Constraints

### C1 — EV port current bound
```
I_ev(j, t) ≤ I_max_j        ∀ j ∈ {1,…,J},  t ∈ window
```

### C2 — BESS current bounds (de-rated)
```
I_bess_ch(t)  ≤ r_bess_ch(t)  · I_high     ∀ t
I_bess_dis(t) ≤ r_bess_dis(t) · I_low      ∀ t
```

### C3 — Grid power cap
```
Σ_j  V_j · I_ev(j,t)  +  V_{J+1} · (I_bess_ch(t) − I_bess_dis(t))  +  L(t)  ≤  P_max     ∀ t
```
Background load `L(t)` is treated as zero (conservative: grid cap is never tighter than it should be).

### C4 — BESS SoC dynamics
```
SoCB(t_start) = SoCB_init  +  (I_bess_ch(t_start) − I_bess_dis(t_start)) · V_{J+1} · δt/1000

SoCB(t) = SoCB(t−1)  +  (I_bess_ch(t) − I_bess_dis(t)) · V_{J+1} · δt/1000     ∀ t > t_start
```
Efficiency appears only in the objective cost terms, not in the SoC update, matching the Chargax simulator's internal BESS model.

### C5 — BESS SoC bounds
```
SoCB_min ≤ SoCB(t) ≤ SoCB_max     ∀ t
```

### C6 — EV SoC dynamics
```
soc_car(i, t_start) = soc_now_i  +  I_ev(y_i, t_start) · V_{y_i} · δt/1000

soc_car(i, t) = soc_car(i, t−1)  +  I_ev(y_i, t) · V_{y_i} · δt/1000     t_start < t ≤ dep_i

soc_car(i, t) = soc_car(i, t−1)                                              t > dep_i
```
where `y_i` is the port assigned to EV `i` (precomputed by Chargax's FCFS scheduler).

### C7 — EV SoC bounds
```
s_min_i ≤ soc_car(i, t) ≤ s_cap_i     ∀ i, t
```

### C8 — EV charging power limit (SoC-dependent de-rating)
```
I_ev(y_i, t) · V_{y_i} ≤ r_car(i, t) · P_car_max_i     ∀ i,  arr_i ≤ t ≤ dep_i
```

### C9 — Occupancy enforcement
```
I_ev(j, t) = 0     if z(j, t) = 0
```
No current may flow on an unoccupied port. This is encoded as a fixed equality (not a conditional big-M) because `z` is a precomputed parameter, keeping the problem a pure LP.

### C10 — Must-serve constraints (MPC only, unless disabled)

These prevent the rolling LP from procrastinating — deferring charging to the end of each window and repeating on every re-solve.

**Pacing floor** (enforced at every re-solve step `t_start`):
```
soc_car(i, t_start) ≥ soc_now_i  +  required_kwh_i
```
where:
```
required_kwh_i = min(
    needed_i / steps_until_dep_i,          # linear pacing rate
    P_max · δt / (n_cars · 1000) · 0.99,  # grid-share budget (1% slack)
    I_max_{y_i} · V_{y_i} · δt / 1000,    # port hardware cap
    needed_i                               # don't require more than needed
)

needed_i = s_target_i − soc_now_i
```

**Hard deadline** (only when `dep_i` falls inside the current window):
```
soc_car(i, dep_i) ≥ s_target_i
```
Skipped if the target is physically unachievable given remaining dwell time.

**Infeasibility fallback:** if C10 + C3 are jointly infeasible (rare, e.g. very tight grid cap with many urgent cars), the controller retries without C10 (`bare=True`) rather than returning no action for the step.

---

## Rolling MPC Loop

```
for each 5-minute step t:
    observe state: soc_now_i, socb_now, assignments, prices p_buy/p_sell over [t, t+H-1]
    build LP over window [t, min(t+H-1, T_max)]
    solve with HiGHS
    apply only the action for step t (I_ev(j,t), I_bess_ch(t), I_bess_dis(t))
    advance to t+1
```

Only the first step of each solution is executed (receding horizon). The LP is re-solved at every step with fresh state observations.

---

## Station Configuration

Default hardware parameters used in experiments:

| Parameter | Value |
|-----------|-------|
| Voltage (EV ports) | 230 V |
| Max current per port | 32 A |
| Grid cap | `n_ports × 7.36 kW` (per-port allocation; scaled by `grid_cap_strain`) |
| BESS voltage | 400 V |
| BESS capacity | 30 kWh |
| BESS max throughput | 10 kW |
| BESS efficiency (one-way) | 0.95 |
| BESS I_high / I_low | 25 A |
| BESS SoC min | 3 kWh |
| Time step | 5 min (δt = 1/12 h) |

**Grid cap strain:** experiments sweep `grid_cap_strain ∈ {1.0, 0.75, 0.5, 0.25}`, multiplying the nominal `P_max` by the strain factor to simulate congested grid connections.

---

## Baselines

The LP controller is benchmarked against two rule-based policies implemented in Chargax:

- **Equal Share:** available grid headroom is divided equally among occupied ports at each step.
- **Max Charge:** all ports charge at the hardware maximum current simultaneously (no coordination).
- **Random:** charges at a uniformly random fraction of the port maximum at each step.

The **clairvoyant offline LP** (`build_ev_lp_model`) solves the full-horizon problem with perfect knowledge of all arrivals and prices, providing an upper bound on achievable profit.
