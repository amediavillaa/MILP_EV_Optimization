from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class StepRecord:
    t: int
    actions: dict           # {port_j: amps}
    reward: float
    profit_delta: float
    served: int
    rejected: int
    step_revenue: float = 0.0
    step_cost: float = 0.0
    compute_ms: float = 0.0


@dataclass
class ChargaxSimResults:
    controller_name: str
    seed: int
    episode_date: str
    net_profit: float
    total_revenue: float
    total_cost: float
    served_customers: int
    rejected_customers: int
    mean_soc_fulfillment: float
    total_compute_s: float = 0.0
    mean_step_ms: float = 0.0
    step_log: list[StepRecord] = field(default_factory=list)

    @classmethod
    def from_final_state(
        cls,
        controller_name: str,
        seed: int,
        state,
        step_log: list[StepRecord],
        departures_fulfillment: list[float],
    ) -> ChargaxSimResults:
        mean_fulfillment = (
            sum(departures_fulfillment) / len(departures_fulfillment)
            if departures_fulfillment else 0.0
        )
        total_revenue = sum(r.step_revenue for r in step_log)
        total_cost = sum(r.step_cost for r in step_log)
        compute_times = [r.compute_ms for r in step_log]
        total_compute_s = sum(compute_times) / 1000.0
        mean_step_ms = sum(compute_times) / len(compute_times) if compute_times else 0.0
        return cls(
            controller_name=controller_name,
            seed=seed,
            episode_date=str(state.datetime),
            net_profit=total_revenue - total_cost,
            total_revenue=total_revenue,
            total_cost=total_cost,
            served_customers=int(state.served_customers),
            rejected_customers=int(state.rejected_customers),
            mean_soc_fulfillment=mean_fulfillment,
            total_compute_s=total_compute_s,
            mean_step_ms=mean_step_ms,
            step_log=step_log,
        )

    def to_dict(self) -> dict:
        return asdict(self)
