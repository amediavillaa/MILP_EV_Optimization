"""
run_chargax_benchmark.py — Run all controllers through Chargax and print summary tables.

Usage examples (see README for more details):
    python -m evopt.experiments.run_chargax_benchmark
    python -m evopt.experiments.run_chargax_benchmark --seeds 3
    python -m evopt.experiments.run_chargax_benchmark --tariff dynamic:1.3
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

import sys
from pathlib import Path

from evopt.analysis.utils import (
    format_experiment_id,
    save_metadata,
    strain_save_path,
    validate_grid_cap_strain,
)

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
    bess_derating:          float = 1.0,
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
                bess_derating=bess_derating,
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
    bess_derating:          float            = 1.0,
    save_path:              str | None       = None,
    cli_command:            str              = "",
    grid_cap_strain_values: list[float]      = None,
) -> None:
    VOLTAGE           = 400.0
    I_MAX             = 32.0
    KW_PER_PORT       = 6.0   # chosen utilization factor per port (~47% of 400V×32A=12.8 kW at current config)
    V_BESS            = 400.0
    P_BESS_MAX_KW     = 10.0
    I_BESS            = P_BESS_MAX_KW * 1000.0 / V_BESS   # 25 A

    if horizons is None:
        horizons = [12]
    if ports is None:
        ports = [3]
    if grid_cap_strain_values is None:
        grid_cap_strain_values = [1.0]
    validate_grid_cap_strain(grid_cap_strain_values)

    tariff_str = (
        f"dynamic:{ev_markup}" if ev_markup is not None
        else "dynamic" if ev_tariff is None
        else str(ev_tariff)
    )
    experiment_id = format_experiment_id(
        ports=ports, horizons=horizons, tariff=tariff_str,
    )

    per_port_raw_dfs:   list[pd.DataFrame] = []

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
            .sort_values(max(grid_cap_strain_values), ascending=False)
        )
        W = 70
        print(f"\n{'=' * W}")
        ports_note = f"  (averaged over ports {ports})" if len(ports) > 1 else ""
        print(f"  Cross-strain summary  (mean net_profit over seeds){ports_note}")
        print(f"{'=' * W}")
        print(strain_pivot.to_string())
        print()

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
    parser.add_argument(
        "--bess-derating", type=float, default=1.0,
        help="BESS charge/discharge capacity derating factor in [0,1] (default: 1.0 = no derating)",
    )
    parser.add_argument(
        "--grid-cap-strain", type=float, nargs="+", default=[1.0],
        metavar="STRAIN",
        help="Grid cap multiplier(s) in (0, 1]. 1.0 = original cap, 0.5 = half cap. "
             "Multiple values run a sweep, e.g. --grid-cap-strain 1.0 0.75 0.5 0.25",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Save raw per-seed results as CSV to PATH (e.g. results/benchmark.csv). "
             "A metadata.json sidecar is written to the same directory.",
    )
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
        bess_derating=args.bess_derating,
        save_path=args.save,
        cli_command=" ".join(sys.argv),
        grid_cap_strain_values=args.grid_cap_strain,
    )
