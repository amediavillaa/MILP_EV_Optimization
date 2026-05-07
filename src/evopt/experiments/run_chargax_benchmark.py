"""
run_chargax_benchmark.py — Run all controllers through Chargax and print a summary table.

Usage:
    python -m evopt.experiments.run_chargax_benchmark
    python -m evopt.experiments.run_chargax_benchmark --seeds 3
    python -m evopt.experiments.run_chargax_benchmark --tariff 0.75          # static €/kWh
    python -m evopt.experiments.run_chargax_benchmark --tariff dynamic        # live Chargax p_sell
    python -m evopt.experiments.run_chargax_benchmark --horizons 1 6 12 24   # compare horizons
"""
from __future__ import annotations

import argparse

from chargax import Chargax

from evopt.benchmarking.results import ChargaxSimResults
from evopt.benchmarking.runner import BenchmarkRunner
from evopt.benchmarking.storage import build_summary_from_results
from evopt.controllers.chargax_baselines import MaxChargeController, RandomController
from evopt.controllers.equal_share import EqualShareController
from evopt.controllers.lp_controller import LPController
from evopt.env.chargax_wrapper import ChargaxWrapper
from evopt.experiments.station_configs import build_station_with_battery


def _parse_tariff(raw: str) -> float | None:
    """Return float for a fixed tariff or None for dynamic (live p_sell)."""
    if raw.lower() == "dynamic":
        return None
    return float(raw)


def main(
    n_seeds: int = 10,
    ev_tariff: float | None = 0.75,
    horizons: list[int] | None = None,
) -> None:
    N_PORTS       = 3
    VOLTAGE       = 400.0
    I_MAX         = 32.0
    P_MAX_KW      = 20.0
    V_BESS        = 400.0
    P_BESS_MAX_KW = 10.0
    I_BESS        = P_BESS_MAX_KW * 1000.0 / V_BESS   # 25 A

    if horizons is None:
        horizons = [12]

    print(f"EV tariff : {'dynamic (Chargax p_sell)' if ev_tariff is None else f'{ev_tariff} €/kWh'}")
    print(f"Horizons  : {horizons}")
    print(f"Seeds     : {n_seeds}\n")

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
        ev_tariff=ev_tariff,
        v_bess=V_BESS,
        I_high=I_BESS,
        I_low=I_BESS,
        socb_min=3.0,
        socb_max=30.0,
        p_bess_max_kw=P_BESS_MAX_KW,
    )

    controllers = {
        **{
            f"milp_h{h}": LPController(
                horizon_steps=h, solver="highs",
                v_bess=V_BESS, I_high=I_BESS, I_low=I_BESS,
                socb_min=3.0, socb_max=30.0,
            )
            for h in horizons
        },
        "equal_share": EqualShareController(),
        "max_charge":  MaxChargeController(),
        "random":      RandomController(seed=0),
    }

    runner  = BenchmarkRunner(env, wrapper)
    results: list[ChargaxSimResults] = []
    seeds   = list(range(n_seeds))
    for name, ctrl in controllers.items():
        for seed in seeds:
            print(f"  {name}  seed={seed} …", end="\r")
            results.append(runner.run_episode(ctrl, seed, name=name))
    print()

    df = build_summary_from_results(results)
    summary = (
        df.groupby("controller")[[
            "net_profit", "total_revenue", "total_cost",
            "served_customers", "rejected_customers",
            "mean_soc_at_departure", "gap_to_best",
        ]]
        .mean()
        .round({"net_profit": 2, "total_revenue": 2, "total_cost": 2,
                "served_customers": 1, "rejected_customers": 1,
                "mean_soc_at_departure": 3, "gap_to_best": 2})
        .sort_values("net_profit", ascending=False)
    )
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, default=10)
    parser.add_argument("--tariff",   type=str, default="0.75",
                        help="Fixed EV tariff in €/kWh (e.g. 0.75) or 'dynamic' for live Chargax p_sell")
    parser.add_argument("--horizons", type=int, nargs="+", default=[12],
                        help="One or more LP horizon lengths in steps (e.g. --horizons 1 6 12 24)")
    args = parser.parse_args()
    main(
        n_seeds=args.seeds,
        ev_tariff=_parse_tariff(args.tariff),
        horizons=args.horizons,
    )
