"""
run_horizon_comparison.py — Compare LPController performance across horizon lengths.

Usage:
    python -m milp_ev_opt.experiments.run_horizon_comparison
    python -m milp_ev_opt.experiments.run_horizon_comparison --seeds 5 --output results/horizon
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chargax import Chargax

from milp_ev_opt.benchmarking.runner import BenchmarkRunner
from milp_ev_opt.benchmarking.storage import build_summary
from milp_ev_opt.controllers.lp_controller import LPController
from milp_ev_opt.env.chargax_wrapper import ChargaxWrapper
from milp_ev_opt.experiments.station_configs import build_station_with_battery


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
        allow_discharging=False,
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
