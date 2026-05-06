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
