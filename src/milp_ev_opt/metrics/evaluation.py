from __future__ import annotations

import statistics
from milp_ev_opt.benchmarking.results import ChargaxSimResults


def compute_summary_metrics(results: list[ChargaxSimResults]) -> dict:
    """
    Group results by controller and return mean ± std for key metrics.

    Returns
    -------
    dict[controller_name, dict[metric, value]]
    """
    grouped: dict[str, list[ChargaxSimResults]] = {}
    for r in results:
        grouped.setdefault(r.controller_name, []).append(r)

    summary = {}
    for name, rs in grouped.items():
        profits      = [r.net_profit        for r in rs]
        served_rates = [
            r.served_customers / max(1, r.served_customers + r.rejected_customers)
            for r in rs
        ]
        reject_rates = [
            r.rejected_customers / max(1, r.served_customers + r.rejected_customers)
            for r in rs
        ]
        socs = [r.mean_soc_fulfillment for r in rs]

        summary[name] = {
            "net_profit_mean":      statistics.mean(profits),
            "net_profit_std":       statistics.pstdev(profits),
            "served_rate_mean":     statistics.mean(served_rates),
            "served_rate_std":      statistics.pstdev(served_rates),
            "rejection_rate_mean":  statistics.mean(reject_rates),
            "rejection_rate_std":   statistics.pstdev(reject_rates),
            "mean_soc_fulfillment": statistics.mean(socs),
        }
    return summary
