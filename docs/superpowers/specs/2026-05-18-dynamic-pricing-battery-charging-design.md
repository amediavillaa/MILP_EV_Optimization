# Design: Fix Dynamic Pricing and BESS Charging in LP Models

**Date:** 2026-05-18  
**Status:** Approved

---

## Problem

Two bugs cause the LP controller to under-use BESS and report inconsistent profits.

### Bug 1 — BESS discharge mispriced in the objective

In `objective.py`, BESS discharge earns `bess_sell_revenue = I_bess_dis × V_bess × p_sell_bess × Δt`.  
`p_sell_bess` is set to the market sell price (from Chargax `future_sell_prices`).

But in Chargax, BESS discharge physically **reduces the station's net grid draw** — it does not sell energy back to the external grid. The C3 grid-cap constraint already models this correctly:

```
ev_power + (I_bess_ch − I_bess_dis) × V_bess ≤ P_max
```

The consequence: BESS discharge is valued at `p_sell_bess` in the objective, but it actually saves `p_buy` per kWh (avoided grid purchase). Since `p_buy > p_sell_bess` in all realistic market conditions, the LP systematically undervalues BESS discharge and rarely chooses to use it.

### Bug 2 — Runner ignores BESS in step-level profit accounting

In `runner.py`, `step_cost` and `step_revenue` are accumulated only over EV charging flows. BESS charging costs and discharge savings are never added. This causes `total_revenue − total_cost` to diverge from Chargax's authoritative `state.profit`, which sees the full station energy balance.

---

## Design

### Section 1 — Corrected objective (`objective.py` + `model.py`)

Replace the three-term formulation:

```
profit = ev_revenue + bess_sell_revenue − grid_cost
```

with a two-term formulation:

```
profit = ev_revenue − grid_cost
```

where `grid_cost` is redefined to use the full station net draw:

```
grid_cost = Σ_t (ev_power_t + V_bess·I_bess_ch_t − V_bess·I_bess_dis_t) · p_buy_t · Δt/1000
```

BESS discharge is now priced at the avoided buy cost (`p_buy`), which is the correct arbitrage signal. This change applies identically to `add_offline_profit_objective` and `add_rolling_profit_objective`.

The `m.p_sell_bess` Param is removed from both model builders in `model.py` — it was only consumed by the objective and is now unused.

### Section 2 — Remove dead `p_sell_bess` infrastructure

With the objective fixed, `p_sell_bess` has no consumer anywhere in the pipeline. Remove it from:

| File | Change |
|------|--------|
| `optimization/model.py` | Remove `m.p_sell_bess = Param(...)` from `build_ev_lp_model` and `build_rolling_model`; remove from docstring |
| `controllers/lp_controller.py` | Remove `"p_sell_bess"` key from `_build_lp_data` |
| `env/chargax_wrapper.py` | Remove `p_sell_bess` dict construction and `"p_sell_bess"` from returned state dict; remove dead comment |

No behaviour changes — pure deletions.

### Section 3 — Fix runner profit tracking (`runner.py`)

After the existing per-car loop, add the BESS net grid draw contribution to `step_cost`:

```python
bess_port = clean_state["J"] + 1
bess_net_amps = actions.get(bess_port, 0.0)   # LP convention: positive = discharging
if self.wrapper.v_bess is not None and bess_net_amps != 0.0:
    bess_net_kwh = -bess_net_amps * self.wrapper.v_bess * delta_t / 1000.0
    # positive → BESS charging (increases cost); negative → BESS discharging (reduces cost)
    step_cost += bess_net_kwh * clean_state["p_buy"].get(t, 0.0)
```

`self.wrapper.v_bess` is already accessible on the runner. No new state keys or data-dict fields are required.

**Expected outcome:** `total_revenue − total_cost` matches `state.profit` (Chargax's authoritative value) once BESS operations are included in both.

---

## Files Changed

| File | Change type |
|------|-------------|
| `src/evopt/optimization/objective.py` | Fix: remove `bess_sell_revenue`; update `grid_cost` formula |
| `src/evopt/optimization/model.py` | Remove: `m.p_sell_bess` Param from both model builders |
| `src/evopt/controllers/lp_controller.py` | Remove: `"p_sell_bess"` from `_build_lp_data` |
| `src/evopt/env/chargax_wrapper.py` | Remove: `p_sell_bess` dict and state key |
| `src/evopt/benchmarking/runner.py` | Fix: add BESS net energy to `step_cost` |

---

## Out of Scope

- BESS round-trip efficiency modelling (Chargax uses 0.95; LP currently assumes 1.0) — separate improvement
- Dynamic pricing urgency-term interaction — the `_URGENCY_EPS` term is negligible (~0.01% of `p_sell`) and left unchanged
- `p_sell_bess` support for a future "BESS sells to external grid" mode — not modelled in Chargax
