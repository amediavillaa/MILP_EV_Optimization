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
from evopt.analysis.utils import strain_save_path, validate_grid_cap_strain
from evopt.experiments.station_configs import build_station_with_battery

VOLTAGE       = 400.0
I_MAX         = 32.0
KW_PER_PORT   = 6.0
V_BESS        = 400.0
P_BESS_MAX_KW = 10.0
I_BESS        = P_BESS_MAX_KW * 1000.0 / V_BESS  # 25 A
EV_TARIFF     = 0.75


def _collect_scenario(env, wrapper: ChargaxWrapper, seed: int) -> dict:
    """Pass 1: run MaxCharge controller to collect full episode scenario.

    Uses MaxCharge (not null) so cars depart on schedule and new customers
    can arrive, matching the episode structure of the MPC replay pass.
    """
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key)
    wrapper.reset()
    collector = ScenarioCollector()
    charge_ctrl = MaxChargeController()
    charge_ctrl.reset()

    done = False
    while not done:
        clean_state = wrapper.extract_state(obs, state)
        collector.record(clean_state)

        actions = charge_ctrl.compute_action(clean_state)
        chargax_actions = wrapper.to_chargax_actions(actions)
        key, subkey = jax.random.split(key)
        timestep, state = env.step_env(subkey, state, chargax_actions)
        obs = timestep.observation
        done = bool(timestep.terminated) or bool(timestep.truncated)

    return collector.build_scenario()


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
        batt_capacity_kwh=30.0, batt_max_kw=P_BESS_MAX_KW, batt_efficiency=0.95,
    )
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=False,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=n_ports, v=VOLTAGE, i_max=I_MAX, p_max_kw=P_MAX_KW_adj,
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
                "grid_cap_strain":  grid_cap_strain,
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
            "grid_cap_strain":  grid_cap_strain,
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
                strain_df = df[(df["grid_cap_strain"] - strain).abs() < 1e-9]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="?", default=10)
    parser.add_argument("--horizons",type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--ports",   type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--save",    type=str, default=None)
    parser.add_argument(
        "--grid-cap-strain", type=float, nargs="+", default=[1.0],
        metavar="STRAIN",
        help="Grid cap multiplier(s) in (0, 1]. 1.0 = original cap, 0.5 = half cap.",
    )
    args = parser.parse_args()
    main(
        n_seeds=args.seeds,
        horizons=args.horizons,
        ports=args.ports,
        save_path=args.save,
        grid_cap_strain_values=args.grid_cap_strain,
    )
