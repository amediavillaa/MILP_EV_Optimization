# Grid-Cap Strain Flag — Design Spec

**Date:** 2026-06-11
**Status:** Approved

## Purpose

Add a `--grid-cap-strain` flag to the experiment pipeline to stress-test the LP controller under tighter grid-cap conditions. The flag scales the station grid power cap `P_max` before each experiment run, enabling a grid-cap stress-test table that compares LP profit against MaxCharge and EqualShare baselines at different levels of grid constraint.

Target output:

| Grid cap strain | LP profit | MaxCharge profit | EqualShare profit | LP gain vs MaxCharge | LP gain vs EqualShare |
|---:|---:|---:|---:|---:|---:|
| 1.00 | ... | ... | ... | ... | ... |
| 0.75 | ... | ... | ... | ... | ... |
| 0.50 | ... | ... | ... | ... | ... |
| 0.25 | ... | ... | ... | ... | ... |

---

## Definition

```
P_max_adjusted = P_max_original * grid_cap_strain
```

- `grid_cap_strain = 1.0` — original grid cap (default, backward-compatible)
- `grid_cap_strain = 0.75` — 75% of original grid cap
- `grid_cap_strain = 0.50` — 50% of original grid cap
- `grid_cap_strain = 0.25` — 25% of original grid cap

---

## Approach: Loop inside existing `main()` (Approach A)

`grid_cap_strain_values: list[float]` (default `[1.0]`) is added to `main()`. The CLI accepts `nargs="+"` so multiple values can be passed in a single invocation. The inner loop over strain values nests inside the existing per-port loop, matching the pattern of `--horizons` and `--ports`.

---

## Section 1 — CLI and Validation

### CLI argument (both scripts)

```
--grid-cap-strain  float [float ...]   default: 1.0
```

Example invocations:
```bash
python -m evopt.experiments.run_chargax_benchmark --grid-cap-strain 0.75
python -m evopt.experiments.run_chargax_benchmark --grid-cap-strain 1.0 0.75 0.5 0.25
python -m evopt.experiments.run_clairvoyant_benchmark --grid-cap-strain 0.5
```

### Validation

Performed once in `main()` before any experiment runs:

```python
for s in grid_cap_strain_values:
    if not (0.0 < s <= 1.0):
        raise ValueError(f"--grid-cap-strain must be in (0, 1]; got {s}")
```

`main()` signature change:

```python
def main(
    ...
    grid_cap_strain_values: list[float] = [1.0],
    ...
) -> None:
```

---

## Section 2 — Applying the Multiplier

### Single source of truth

The multiplier is applied at exactly one point per script — the `P_MAX_KW` computation — before anything is constructed:

```python
P_MAX_KW = n_ports * KW_PER_PORT         # original base cap
P_MAX_KW_adj = P_MAX_KW * grid_cap_strain  # adjusted cap for this run
```

`P_MAX_KW_adj` replaces `P_MAX_KW` in:
- `build_simple_station(p_max_kw=P_MAX_KW_adj)` / `build_station_with_battery(p_max_kw=P_MAX_KW_adj)`
- `ChargaxWrapper(p_max_kw=P_MAX_KW_adj)`

### No changes needed downstream

All controllers and the LP model read `P_max` from the state dict, which `ChargaxWrapper` populates from `self.P_max_w = P_MAX_KW_adj * 1000`:

- `EqualShareController` — reads `state["P_max"]` automatically
- `LPController._build_lp_data()` — reads `state["P_max"]` → Pyomo `m.P_max` → C3 grid cap constraint
- `MaxChargeController`, `RandomController` — request I_max per port; wrapper's `to_chargax_actions` hard-clips to `P_max_w`
- `BenchmarkRunner.run_episode()` — reads `clean_state["P_max"]` for the grid cap scaling factor
- `add_rolling_constraints()` / `add_offline_constraints()` — use `m.P_max` loaded from data dict

No changes to: `constraints.py`, `model.py`, `lp_controller.py`, `equal_share.py`, `chargax_baselines.py`, `chargax_wrapper.py`, `station_configs.py`.

---

## Section 3 — Experiment ID, Filenames, and Output Directories

### `format_experiment_id()` in `utils.py`

Gains an optional `grid_cap_strain: float = 1.0` kwarg. When strain is not 1.0, appends `_s<safe_value>`:

```python
# strain=1.0  → "0_75_fixed_0_75_p3_h12"         (unchanged)
# strain=0.75 → "0_75_fixed_0_75_p3_h12_s0_75"
# strain=0.50 → "0_75_fixed_0_75_p3_h12_s0_50"
```

### Output directory per strain

When `--save` is used with a single strain value of 1.0, the path is unchanged (backward-compatible). When multiple strain values are provided, the script inserts `_s<strain>` before the `.csv` suffix:

```
# Single value (default) — path unchanged
--save results/benchmark.csv  +  strain [1.0]
→ results/benchmark.csv

# Multiple values — per-strain subdirectories
--save results/benchmark.csv  +  strains [1.0, 0.75, 0.5]
→ results/benchmark_s1_00/benchmark.csv
→ results/benchmark_s0_75/benchmark.csv
→ results/benchmark_s0_50/benchmark.csv
```

Each strain level also gets its own `metadata.json` sidecar in the same subdirectory.

---

## Section 4 — Result Tables and Metadata

### CSV column

The per-seed CSV gains one new column: `grid_cap_strain` (float). Existing columns are unchanged — backward-compatible for any existing analysis.

### `save_metadata()` in `utils.py`

Gains `grid_cap_strain` as an optional key (default `1.0`), written to `metadata.json` alongside `ports`, `horizons`, etc.

### Terminal output

A cross-strain pivot table is printed at the end of any multi-strain run, showing `net_profit` by `(controller, grid_cap_strain)`. Printed only when `len(grid_cap_strain_values) > 1`, analogous to the existing cross-port summary.

---

## Section 5 — Tests

New file: `tests/test_grid_cap_strain.py`

### Test 1 — Validation rejects out-of-range input
- `grid_cap_strain = 0.0` raises `ValueError`
- `grid_cap_strain = 1.1` raises `ValueError`
- `grid_cap_strain = 1.0` and `0.5` do not raise

### Test 2 — Multiplier is applied correctly
- Given `P_MAX_KW = 18.0` and `grid_cap_strain = 0.5`, `P_MAX_KW_adj = 9.0`
- A `ChargaxWrapper` built with `p_max_kw=9.0` stores `P_max_w = 9000.0`
- A state dict extracted from this wrapper has `state["P_max"] == 9000.0`

### Test 3 — EqualShare respects tighter cap (integration)
- Build two `ChargaxWrapper` instances: one at strain=1.0 (P_max=18 kW) and one at strain=0.5 (P_max=9 kW)
- Construct a synthetic state dict with 3 cars, pass to `EqualShareController.compute_action()`
- Assert that the total allocated power at strain=0.5 is ≤ half that at strain=1.0
- No full Chargax episode needed — tests the wrapper + controller in isolation

---

## Files Changed

| File | Change |
|---|---|
| `src/evopt/experiments/run_chargax_benchmark.py` | Add `--grid-cap-strain` CLI arg, `grid_cap_strain_values` loop, adjusted `P_MAX_KW_adj`, strain column in CSV, cross-strain summary table |
| `src/evopt/experiments/run_clairvoyant_benchmark.py` | Same CLI arg and multiplier application |
| `src/evopt/analysis/utils.py` | `format_experiment_id()` gains `grid_cap_strain` kwarg; `save_metadata()` gains `grid_cap_strain` field |
| `tests/test_grid_cap_strain.py` | New — 3 tests as described above |

---

## Backward Compatibility

- Default `grid_cap_strain_values = [1.0]` — existing invocations without the flag behave identically
- Existing CSV files are unaffected; new runs add one column
- `format_experiment_id()` signature uses a kwarg with default, so callers that omit it get the same ID as before
