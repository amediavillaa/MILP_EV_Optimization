# Grid-Cap Strain Flag — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--grid-cap-strain` to both experiment benchmark scripts, scaling `P_max` before station and wrapper construction so all controllers (LP, EqualShare, MaxCharge, Random) see the adjusted grid cap automatically.

**Architecture:** The multiplier is applied at the single `P_MAX_KW` computation point in each script, before the Chargax station and `ChargaxWrapper` are built. Downstream code — controllers, LP model, and runner — reads `P_max` from the state dict provided by the wrapper and requires no changes. Shared utilities (`validate_grid_cap_strain`, `strain_save_path`, and extensions to `format_experiment_id` / `save_metadata`) live in `src/evopt/analysis/utils.py` and are imported by both scripts.

**Tech Stack:** Python, argparse, pandas — no new dependencies.

---

## File Map

| Path | Status | Responsibility |
|---|---|---|
| `tests/test_grid_cap_strain.py` | Create | 5 tests: validation (3), multiplier property (1), EqualShare integration (1) |
| `src/evopt/analysis/utils.py` | Modify | Add `validate_grid_cap_strain`, `strain_save_path`; extend `format_experiment_id`, `save_metadata` |
| `src/evopt/experiments/run_chargax_benchmark.py` | Modify | CLI arg, validation, strain loop, adjusted cap, CSV column, summary table, save paths |
| `src/evopt/experiments/run_clairvoyant_benchmark.py` | Modify | CLI arg, validation, strain loop, adjusted cap, CSV column, save paths |

---

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_grid_cap_strain.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_grid_cap_strain.py
import pytest


# ── Validation ────────────────────────────────────────────────────────────────

def test_validation_rejects_zero():
    from evopt.analysis.utils import validate_grid_cap_strain
    with pytest.raises(ValueError, match="must be in"):
        validate_grid_cap_strain([0.0])


def test_validation_rejects_above_one():
    from evopt.analysis.utils import validate_grid_cap_strain
    with pytest.raises(ValueError, match="must be in"):
        validate_grid_cap_strain([1.1])


def test_validation_accepts_valid():
    from evopt.analysis.utils import validate_grid_cap_strain
    validate_grid_cap_strain([1.0])
    validate_grid_cap_strain([0.5])
    validate_grid_cap_strain([1.0, 0.75, 0.5, 0.25])


# ── Multiplier property ────────────────────────────────────────────────────────

def test_multiplier_applied_to_wrapper():
    from evopt.env.chargax_wrapper import ChargaxWrapper
    wrapper = ChargaxWrapper(n_ports=3, v=400.0, i_max=32.0, p_max_kw=9.0)
    assert wrapper.P_max_w == pytest.approx(9000.0)


# ── EqualShare integration ─────────────────────────────────────────────────────

def _make_state(p_max_kw: float) -> dict:
    P_max_w = p_max_kw * 1000.0
    return {
        "t": 0,
        "delta_t": 5 / 60,
        "J": 3,
        "P_max": P_max_w,
        "V":     {1: 400.0, 2: 400.0, 3: 400.0},
        "I_max": {1: 32.0,  2: 32.0,  3: 32.0},
        "p_buy":  {0: 0.10},
        "p_sell": {0: 0.75},
        "present_cars": {
            0: {"soc_now": 5.0, "s_target": 20.0, "s_cap": 60.0, "t_max": 20},
            1: {"soc_now": 5.0, "s_target": 20.0, "s_cap": 60.0, "t_max": 20},
            2: {"soc_now": 5.0, "s_target": 20.0, "s_cap": 60.0, "t_max": 20},
        },
        "assignments": {0: 1, 1: 2, 2: 3},
        "departed_fulfillments": [],
    }


def test_equal_share_respects_tighter_cap():
    from evopt.controllers.equal_share import EqualShareController
    ctrl = EqualShareController()

    state_full = _make_state(18.0)   # strain = 1.0  (3 ports × 6 kW)
    state_half = _make_state(9.0)    # strain = 0.5

    actions_full = ctrl.compute_action(state_full)
    actions_half = ctrl.compute_action(state_half)

    V = 400.0
    power_full = sum(a * V for a in actions_full.values())
    power_half = sum(a * V for a in actions_half.values())

    # EqualShare splits P_max/N per car; halving P_max must halve total power
    assert power_half <= power_full * 0.5 + 1e-6
```

- [ ] **Step 2: Run tests and confirm the right ones fail**

```
pytest tests/test_grid_cap_strain.py -v
```

Expected: `test_validation_*` (3 tests) FAIL with `ImportError` — `validate_grid_cap_strain` not yet in `utils.py`. `test_multiplier_applied_to_wrapper` and `test_equal_share_respects_tighter_cap` PASS — they only use existing code.

---

### Task 2: Extend `utils.py`

**Files:**
- Modify: `src/evopt/analysis/utils.py`

- [ ] **Step 1: Add `validate_grid_cap_strain` and `strain_save_path`**

Append these two functions at the end of `src/evopt/analysis/utils.py`:

```python
def validate_grid_cap_strain(values: list[float]) -> None:
    for s in values:
        if not (0.0 < s <= 1.0):
            raise ValueError(
                f"--grid-cap-strain must be in (0, 1]; got {s}"
            )


def strain_save_path(save_path: str, strain: float) -> Path:
    """Return a per-strain subdirectory path.

    'results/bench.csv', 0.75  →  'results/bench_s0_75/bench.csv'
    """
    p = Path(save_path)
    safe = f"{strain:.2f}".replace(".", "_")
    return p.parent / f"{p.stem}_s{safe}" / p.name
```

- [ ] **Step 2: Add `grid_cap_strain` kwarg to `format_experiment_id`**

Replace the existing `format_experiment_id` function (lines 53–63):

```python
def format_experiment_id(
    ports: list[int],
    horizons: list[int],
    tariff: str,
    grid_cap_strain: float = 1.0,
    **_kwargs,
) -> str:
    """Build a human-readable experiment ID, e.g. 'dynamic_1_3_p3_6_h1_6'."""
    safe_tariff = re.sub(r"[^a-zA-Z0-9]", "_", tariff)
    p_str = "_".join(str(p) for p in sorted(ports))
    h_str = "_".join(str(h) for h in sorted(horizons))
    base = f"{safe_tariff}_p{p_str}_h{h_str}"
    if grid_cap_strain == 1.0:
        return base
    safe_strain = f"{grid_cap_strain:.2f}".replace(".", "_")
    return f"{base}_s{safe_strain}"
```

- [ ] **Step 3: Add `grid_cap_strain` to `save_metadata`**

In `save_metadata`, add one line to the `meta` dict immediately after `"solver"`:

```python
        "solver":          config.get("solver", ""),
        "grid_cap_strain": config.get("grid_cap_strain", 1.0),
```

- [ ] **Step 4: Run the three validation tests**

```
pytest tests/test_grid_cap_strain.py::test_validation_rejects_zero tests/test_grid_cap_strain.py::test_validation_rejects_above_one tests/test_grid_cap_strain.py::test_validation_accepts_valid -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```
pytest tests/ -v
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/evopt/analysis/utils.py tests/test_grid_cap_strain.py
git commit -m "feat: add validate_grid_cap_strain and strain_save_path; extend format_experiment_id and save_metadata"
```

---

### Task 3: Update `run_chargax_benchmark.py`

**Files:**
- Modify: `src/evopt/experiments/run_chargax_benchmark.py`

- [ ] **Step 1: Add `Path` import and extend the `utils` import**

Replace:
```python
from evopt.analysis.utils import format_experiment_id, save_metadata
```
With:
```python
from pathlib import Path

from evopt.analysis.utils import (
    format_experiment_id,
    save_metadata,
    strain_save_path,
    validate_grid_cap_strain,
)
```

- [ ] **Step 2: Update `main()` signature**

Replace the existing `def main(` signature block with:

```python
def main(
    n_seeds:                int              = 10,
    ev_tariff:              float | None     = 0.75,
    ev_markup:              float | None     = None,
    horizons:               list[int] | None = None,
    ports:                  list[int] | None = None,
    use_bess:               bool             = True,
    allow_discharging:      bool             = False,
    allow_bess_discharging: bool             = False,
    must_serve:             bool             = True,
    bess_derating:          float            = 1.0,
    save_path:              str | None       = None,
    cli_command:            str              = "",
    grid_cap_strain_values: list[float]      = None,
) -> None:
```

- [ ] **Step 3: Add default and validation at the start of `main()` body**

After the existing `if horizons is None` / `if ports is None` defaults, add:

```python
    if grid_cap_strain_values is None:
        grid_cap_strain_values = [1.0]
    validate_grid_cap_strain(grid_cap_strain_values)
```

- [ ] **Step 4: Replace the per-port loop with the strain inner loop**

Replace the entire `per_port_raw_dfs: list[pd.DataFrame] = []` block through to `per_port_summaries.append(summary)` with:

```python
    per_port_raw_dfs:   list[pd.DataFrame] = []
    per_port_summaries: list[pd.DataFrame] = []

    for n_ports in ports:
        P_MAX_KW = n_ports * KW_PER_PORT
        for grid_cap_strain in grid_cap_strain_values:
            P_MAX_KW_adj = P_MAX_KW * grid_cap_strain
            strain_id = format_experiment_id(
                ports=[n_ports], horizons=horizons, tariff=tariff_str,
                grid_cap_strain=grid_cap_strain,
            )
            W = 70
            strain_label = (
                f"  (grid cap strain = {grid_cap_strain:.2f})"
                if grid_cap_strain != 1.0 else ""
            )
            print(f"\n{'=' * W}")
            print(f"  Ports = {n_ports}  |  grid cap = {P_MAX_KW_adj:.1f} kW"
                  f"  |  I_max = {I_MAX} A{strain_label}")
            print(f"{'=' * W}")

            df = _run_one_config(
                n_ports=n_ports, n_seeds=n_seeds, horizons=horizons,
                ev_tariff=ev_tariff, ev_markup=ev_markup, voltage=VOLTAGE, i_max=I_MAX,
                p_max_kw=P_MAX_KW_adj, v_bess=V_BESS, i_bess=I_BESS,
                p_bess_max_kw=P_BESS_MAX_KW, use_bess=use_bess,
                allow_discharging=allow_discharging,
                allow_bess_discharging=allow_bess_discharging,
                must_serve=must_serve,
                bess_derating=bess_derating,
            )
            raw = df.copy()
            raw["ports"]            = n_ports
            raw["grid_cap_strain"]  = grid_cap_strain
            raw["experiment_id"]    = strain_id
            raw["tariff"]           = tariff_str
            raw["bess_enabled"]     = use_bess
            raw["v2g_enabled"]      = allow_discharging and use_bess
            raw["solver"]           = "highs"
            per_port_raw_dfs.append(raw)

            summary = _make_summary(df)
            print(summary.to_string())

            summary = summary.copy()
            summary["ports"]           = n_ports
            summary["grid_cap_strain"] = grid_cap_strain
            per_port_summaries.append(summary)
```

- [ ] **Step 5: Replace the post-loop summary block**

Replace the existing `if len(ports) > 1:` block with:

```python
    if len(ports) > 1 and len(grid_cap_strain_values) == 1:
        combined = pd.concat(per_port_summaries).reset_index()
        pivot = (
            combined
            .pivot_table(
                values=["net_profit", "served_customers", "rejected_customers"],
                index="controller",
                columns="ports",
            )
            .round({"net_profit": 2, "served_customers": 1, "rejected_customers": 1})
            .sort_values(("net_profit", ports[-1]), ascending=False)
        )
        W = 70
        print(f"\n{'=' * W}")
        print("  Cross-port summary  (mean over seeds)")
        print(f"{'=' * W}")
        print(pivot.to_string())
        print()

    if len(grid_cap_strain_values) > 1:
        combined_strain = pd.concat(per_port_raw_dfs, ignore_index=True)
        strain_pivot = (
            combined_strain
            .groupby(["controller", "grid_cap_strain"])["net_profit"]
            .mean()
            .round(2)
            .unstack("grid_cap_strain")
            .sort_values(grid_cap_strain_values[0], ascending=False)
        )
        W = 70
        print(f"\n{'=' * W}")
        print("  Cross-strain summary  (mean net_profit over seeds)")
        print(f"{'=' * W}")
        print(strain_pivot.to_string())
        print()
```

- [ ] **Step 6: Replace the `if save_path is not None:` block**

Replace the existing save block (everything under `if save_path is not None:`) with:

```python
    if save_path is not None:
        combined_raw = pd.concat(per_port_raw_dfs, ignore_index=True)
        multi_strain = len(grid_cap_strain_values) > 1

        if multi_strain:
            for strain in grid_cap_strain_values:
                strain_df = combined_raw[combined_raw["grid_cap_strain"] == strain]
                strain_p  = strain_save_path(save_path, strain)
                strain_p.parent.mkdir(parents=True, exist_ok=True)
                strain_df.to_csv(strain_p, index=False)
                print(f"\nResults saved to {strain_p}")
                save_metadata(
                    config={
                        "experiment_id":   format_experiment_id(
                            ports=ports, horizons=horizons, tariff=tariff_str,
                            grid_cap_strain=strain,
                        ),
                        "cli_command":     cli_command,
                        "ports":           ports,
                        "horizons":        horizons,
                        "n_seeds":         n_seeds,
                        "tariff":          tariff_str,
                        "bess_enabled":    use_bess,
                        "v2g_enabled":     allow_discharging and use_bess,
                        "solver":          "highs",
                        "grid_cap_strain": strain,
                    },
                    output_dir=strain_p.parent,
                )
        else:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            combined_raw.to_csv(save_p, index=False)
            print(f"\nResults saved to {save_p}")
            save_metadata(
                config={
                    "experiment_id":   experiment_id,
                    "cli_command":     cli_command,
                    "ports":           ports,
                    "horizons":        horizons,
                    "n_seeds":         n_seeds,
                    "tariff":          tariff_str,
                    "bess_enabled":    use_bess,
                    "v2g_enabled":     allow_discharging and use_bess,
                    "solver":          "highs",
                    "grid_cap_strain": grid_cap_strain_values[0],
                },
                output_dir=save_p.parent,
            )
```

- [ ] **Step 7: Add CLI argument and wire to `main()`**

In the `if __name__ == "__main__":` block, add after the existing `--bess-derating` argument:

```python
    parser.add_argument(
        "--grid-cap-strain", type=float, nargs="+", default=[1.0],
        metavar="STRAIN",
        help="Grid cap multiplier(s) in (0, 1]. 1.0 = original cap, 0.5 = half cap. "
             "Multiple values run a sweep, e.g. --grid-cap-strain 1.0 0.75 0.5 0.25",
    )
```

Update the `main(...)` call to add:

```python
        grid_cap_strain_values=args.grid_cap_strain,
```

- [ ] **Step 8: Run all five new tests**

```
pytest tests/test_grid_cap_strain.py -v
```

Expected: all 5 PASS.

- [ ] **Step 9: Run the full test suite**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/evopt/experiments/run_chargax_benchmark.py
git commit -m "feat: add --grid-cap-strain sweep to run_chargax_benchmark"
```

---

### Task 4: Update `run_clairvoyant_benchmark.py`

**Files:**
- Modify: `src/evopt/experiments/run_clairvoyant_benchmark.py`

- [ ] **Step 1: Add imports**

After the existing `from evopt.env.chargax_wrapper import ChargaxWrapper` import line, add:

```python
from pathlib import Path

from evopt.analysis.utils import strain_save_path, validate_grid_cap_strain
```

- [ ] **Step 2: Add `grid_cap_strain` parameter to `_run_one_port` and apply the multiplier**

Replace the `_run_one_port` signature and the two lines that use `P_MAX_KW`:

Old:
```python
def _run_one_port(
    n_ports:  int,
    n_seeds:  int,
    horizons: list[int],
) -> list[dict]:
    P_MAX_KW = n_ports * KW_PER_PORT
    station  = build_station_with_battery(
        n_ports=n_ports, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW,
```

New:
```python
def _run_one_port(
    n_ports:         int,
    n_seeds:         int,
    horizons:        list[int],
    grid_cap_strain: float = 1.0,
) -> list[dict]:
    P_MAX_KW     = n_ports * KW_PER_PORT
    P_MAX_KW_adj = P_MAX_KW * grid_cap_strain
    station  = build_station_with_battery(
        n_ports=n_ports, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW_adj,
```

Also update the `ChargaxWrapper(...)` call inside `_run_one_port` — change `p_max_kw=P_MAX_KW` to `p_max_kw=P_MAX_KW_adj`.

- [ ] **Step 3: Add `grid_cap_strain` to every record dict**

In the MPC horizon loop, replace:
```python
            records.append({
                "ports":            n_ports,
                "seed":             seed,
                "horizon":          h,
                "offline_profit":   round(offline_profit, 4),
                "mpc_profit":       round(mpc_profit, 4),
                "optimality_gap":   round(gap, 6),
            })
```
With:
```python
            records.append({
                "ports":            n_ports,
                "seed":             seed,
                "horizon":          h,
                "grid_cap_strain":  grid_cap_strain,
                "offline_profit":   round(offline_profit, 4),
                "mpc_profit":       round(mpc_profit, 4),
                "optimality_gap":   round(gap, 6),
            })
```

In the `max_charge` baseline `records.append`, add the same `"grid_cap_strain": grid_cap_strain` key.

- [ ] **Step 4: Replace `main()` with the strain-aware version**

Replace the entire `main()` function with:

```python
def main(
    n_seeds:                int              = 10,
    horizons:               list[int]        = None,
    ports:                  list[int]        = None,
    save_path:              str | None       = None,
    grid_cap_strain_values: list[float]      = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 3, 6, 12]
    if ports is None:
        ports = [3, 6, 12]
    if grid_cap_strain_values is None:
        grid_cap_strain_values = [1.0]
    validate_grid_cap_strain(grid_cap_strain_values)

    all_records = []
    for n_ports in ports:
        for grid_cap_strain in grid_cap_strain_values:
            all_records.extend(
                _run_one_port(n_ports, n_seeds, horizons, grid_cap_strain)
            )

    df = pd.DataFrame(all_records)

    print("\n=== Optimality gap: (offline_profit - mpc_profit) / offline_profit ===")
    summary = (
        df[df["horizon"].notna()]
        .groupby(["ports", "horizon", "grid_cap_strain"])["optimality_gap"]
        .agg(mean="mean", std="std")
        .round(4)
        .reset_index()
    )
    print(summary.to_string(index=False))

    if save_path is not None:
        multi_strain = len(grid_cap_strain_values) > 1
        if multi_strain:
            for strain in grid_cap_strain_values:
                strain_df = df[df["grid_cap_strain"] == strain]
                strain_p  = strain_save_path(save_path, strain)
                strain_p.parent.mkdir(parents=True, exist_ok=True)
                strain_df.to_csv(strain_p, index=False)
                print(f"\nSaved to {strain_p}")
        else:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"\nSaved to {out}")

    return df
```

- [ ] **Step 5: Add CLI argument and wire to `main()`**

In the `if __name__ == "__main__":` block, add after the existing `--save` argument:

```python
    parser.add_argument(
        "--grid-cap-strain", type=float, nargs="+", default=[1.0],
        metavar="STRAIN",
        help="Grid cap multiplier(s) in (0, 1]. 1.0 = original cap, 0.5 = half cap.",
    )
```

Update the `main(...)` call to add:

```python
        grid_cap_strain_values=args.grid_cap_strain,
```

- [ ] **Step 6: Run the full test suite**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/evopt/experiments/run_clairvoyant_benchmark.py
git commit -m "feat: add --grid-cap-strain sweep to run_clairvoyant_benchmark"
```
