"""
run_chargax_benchmark.py — Run all controllers through Chargax and print summary tables.

Usage:
    python -m evopt.experiments.run_chargax_benchmark
    python -m evopt.experiments.run_chargax_benchmark --seeds 3
    python -m evopt.experiments.run_chargax_benchmark --tariff dynamic
    python -m evopt.experiments.run_chargax_benchmark --horizons 1 6 12 24
    python -m evopt.experiments.run_chargax_benchmark --ports 3 6 12
    python -m evopt.experiments.run_chargax_benchmark --ports 3 6 12 --horizons 1 12
    python -m evopt.experiments.run_chargax_benchmark --no-bess
    python -m evopt.experiments.run_chargax_benchmark --allow-discharging
"""
from __future__ import annotations

import argparse

import pandas as pd
from chargax import Chargax

from evopt.benchmarking.results import ChargaxSimResults
from evopt.benchmarking.runner import BenchmarkRunner
from evopt.benchmarking.storage import build_summary_from_results
from evopt.controllers.chargax_baselines import MaxChargeController, RandomController
from evopt.controllers.equal_share import EqualShareController
from evopt.controllers.lp_controller import LPController
from evopt.env.chargax_wrapper import ChargaxWrapper
from evopt.experiments.station_configs import build_simple_station, build_station_with_battery

_METRICS = [
    "net_profit", "total_revenue", "total_cost",
    "served_customers", "rejected_customers",
    "mean_soc_fulfillment", "gap_to_best",
    "total_compute_s", "mean_step_ms",
]
_ROUNDING = {
    "net_profit": 2, "total_revenue": 2, "total_cost": 2,
    "served_customers": 1, "rejected_customers": 1,
    "mean_soc_fulfillment": 3, "gap_to_best": 2,
    "total_compute_s": 2, "mean_step_ms": 2,
}


def _parse_tariff(raw: str) -> tuple[float | None, float | None]:
    """Return (ev_tariff, ev_markup).

    '0.75'         → fixed 0.75 €/kWh
    'dynamic'      → live Chargax p_sell (spot export price)
    'dynamic:1.3'  → p_buy × 1.3 (cost-plus dynamic pricing)
    """
    lower = raw.lower()
    if lower == "dynamic":
        return None, None
    if lower.startswith("dynamic:"):
        return None, float(lower.split(":", 1)[1])
    return float(raw), None


def _run_one_config(
    n_ports:                int,
    n_seeds:                int,
    horizons:               list[int],
    ev_tariff:              float | None,
    ev_markup:              float | None,
    voltage:                float,
    i_max:                  float,
    p_max_kw:               float,
    v_bess:                 float,
    i_bess:                 float,
    p_bess_max_kw:          float,
    use_bess:               bool = True,
    allow_discharging:      bool = False,
    allow_bess_discharging: bool = False,
    must_serve:             bool = True,
) -> pd.DataFrame:
    """Run all controllers for one (n_ports, horizons) config. Returns per-seed DataFrame."""
    if use_bess:
        station = build_station_with_battery(
            n_ports=n_ports, v=voltage, i_max=i_max, p_max_kw=p_max_kw,
            batt_capacity_kwh=30.0, batt_max_kw=p_bess_max_kw, batt_efficiency=0.95,
        )
    else:
        station = build_simple_station(
            n_ports=n_ports, v=voltage, i_max=i_max, p_max_kw=p_max_kw,
        )
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=allow_discharging and use_bess,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=n_ports, v=voltage, i_max=i_max, p_max_kw=p_max_kw,
        num_discretization_levels=10,
        minutes_per_step=5,
        ev_tariff=ev_tariff,
        ev_markup=ev_markup,
        **(dict(
            v_bess=v_bess,
            I_high=i_bess,
            I_low=i_bess,
            socb_min=3.0,
            socb_max=30.0,
            p_bess_max_kw=p_bess_max_kw,
            allow_discharging=allow_discharging,
            allow_bess_discharging=allow_bess_discharging,
        ) if use_bess else {}),
    )
    lp_bess_kwargs = dict(
        v_bess=v_bess, I_high=i_bess, I_low=i_bess, socb_min=3.0, socb_max=30.0,
    ) if use_bess else dict(
        v_bess=v_bess, I_high=0.0, I_low=0.0, socb_min=0.0, socb_max=0.0,
    )
    controllers = {
        **{
            f"milp_h{h}": LPController(
                horizon_steps=h, solver="highs", **lp_bess_kwargs,
                must_serve=must_serve,
            )
            for h in horizons
        },
        "equal_share": EqualShareController(),
        "max_charge":  MaxChargeController(),
        "random":      RandomController(seed=0),
    }

    runner  = BenchmarkRunner(env, wrapper)
    results: list[ChargaxSimResults] = []
    for name, ctrl in controllers.items():
        for seed in range(n_seeds):
            print(f"  ports={n_ports}  {name}  seed={seed} …", end="\r")
            results.append(runner.run_episode(ctrl, seed, name=name))
    print()

    return build_summary_from_results(results)


def _make_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("controller")[_METRICS]
        .mean()
        .round(_ROUNDING)
        .sort_values("net_profit", ascending=False)
    )


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
) -> None:
    VOLTAGE           = 400.0
    I_MAX             = 32.0
    KW_PER_PORT       = 6.0   # grid cap per port: ~47% of installed capacity (400V × 32A = 12.8 kW)
    V_BESS            = 400.0
    P_BESS_MAX_KW     = 10.0
    I_BESS            = P_BESS_MAX_KW * 1000.0 / V_BESS   # 25 A

    if horizons is None:
        horizons = [12]
    if ports is None:
        ports = [3]

    print(f"Ports     : {ports}")
    print(f"Horizons  : {horizons}")
    if ev_tariff is not None:
        tariff_desc = f"{ev_tariff} €/kWh (fixed)"
    elif ev_markup is not None:
        tariff_desc = f"dynamic (p_buy × {ev_markup})"
    else:
        tariff_desc = "dynamic (Chargax p_sell — spot export price)"
    print(f"EV tariff : {tariff_desc}")
    print(f"BESS         : {'enabled' if use_bess else 'disabled'}")
    print(f"BESS disc    : {'enabled' if use_bess and allow_bess_discharging else 'disabled'}")
    print(f"V2G (EV disc): {'enabled' if use_bess and allow_discharging else 'disabled'}")
    print(f"Must-serve   : {'enabled' if must_serve else 'disabled'}")
    print(f"Seeds     : {n_seeds}")

    per_port_summaries: list[pd.DataFrame] = []

    for n_ports in ports:
        P_MAX_KW = n_ports * KW_PER_PORT   # scales with port count for consistent utilization
        W = 70
        print(f"\n{'=' * W}")
        print(f"  Ports = {n_ports}  |  grid cap = {P_MAX_KW} kW  |  I_max = {I_MAX} A")
        print(f"{'=' * W}")

        df = _run_one_config(
            n_ports=n_ports, n_seeds=n_seeds, horizons=horizons,
            ev_tariff=ev_tariff, ev_markup=ev_markup, voltage=VOLTAGE, i_max=I_MAX,
            p_max_kw=P_MAX_KW, v_bess=V_BESS, i_bess=I_BESS,
            p_bess_max_kw=P_BESS_MAX_KW, use_bess=use_bess,
            allow_discharging=allow_discharging,
            allow_bess_discharging=allow_bess_discharging,
            must_serve=must_serve,
        )
        summary = _make_summary(df)
        print(summary.to_string())

        summary = summary.copy()
        summary["ports"] = n_ports
        per_port_summaries.append(summary)

    if len(ports) > 1:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, default=10)
    parser.add_argument("--tariff",   type=str, default="0.75",
                        help="Fixed tariff in €/kWh (e.g. 0.75), 'dynamic' (raw p_sell), "
                             "or 'dynamic:1.3' (p_buy × 1.3 cost-plus pricing)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[12],
                        help="LP horizon lengths in steps (e.g. --horizons 1 6 12 24)")
    parser.add_argument("--ports",    type=int, nargs="+", default=[3],
                        help="Number of EV charging ports (e.g. --ports 3 6 12)")
    parser.add_argument("--no-bess", action="store_true", default=False,
                        help="Disable the on-site BESS entirely (default: BESS enabled)")
    parser.add_argument("--allow-bess-discharging", action="store_true", default=False,
                        help="Allow BESS to discharge in the LP (default: disabled)")
    parser.add_argument("--allow-discharging", action="store_true", default=False,
                        help="Enable V2G: bidirectional EVSE action space (default: disabled)")
    parser.add_argument("--no-must-serve", action="store_true", default=False,
                        help="Disable must-serve constraints in the LP (default: enabled)")
    args = parser.parse_args()
    ev_tariff, ev_markup = _parse_tariff(args.tariff)
    main(
        n_seeds=args.seeds,
        ev_tariff=ev_tariff,
        ev_markup=ev_markup,
        horizons=args.horizons,
        use_bess=not args.no_bess,
        allow_discharging=args.allow_discharging,
        allow_bess_discharging=args.allow_bess_discharging,
        ports=args.ports,
        must_serve=not args.no_must_serve,
    )
