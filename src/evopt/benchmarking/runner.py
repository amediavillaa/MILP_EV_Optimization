from __future__ import annotations

from pathlib import Path

import jax

from evopt.benchmarking import storage
from evopt.benchmarking.results import ChargaxSimResults, StepRecord
from evopt.controllers.base_controller import BaseController
from evopt.env.chargax_wrapper import ChargaxWrapper


class BenchmarkRunner:
    """Runs any BaseController through a Chargax episode and collects results."""

    def __init__(self, env, wrapper: ChargaxWrapper) -> None:
        self.env     = env
        self.wrapper = wrapper

    def run_episode(self, controller: BaseController, seed: int) -> ChargaxSimResults:
        key          = jax.random.PRNGKey(seed)
        obs, state   = self.env.reset_env(key)

        controller.reset()
        self.wrapper.reset()

        step_log:       list[StepRecord] = []
        departures_soc: list[float]      = []
        done = False

        while not done:
            clean_state = self.wrapper.extract_state(obs, state)
            departures_soc.extend(clean_state.get("departed_socs", []))

            actions         = controller.compute_action(clean_state)
            chargax_actions = self.wrapper.to_chargax_actions(actions)

            t       = clean_state["t"]
            delta_t = clean_state["delta_t"]
            step_revenue = 0.0
            step_cost    = 0.0
            for car_id, port_j in clean_state["assignments"].items():
                amps       = actions.get(port_j, 0.0)
                v          = clean_state["V"][port_j]
                energy_kwh = amps * v * delta_t / 1000.0
                step_revenue += energy_kwh * clean_state["p_sell"].get(t, 0.0)
                step_cost    += energy_kwh * clean_state["p_buy"].get(t, 0.0)

            key, subkey = jax.random.split(key)
            timestep, state = self.env.step_env(subkey, state, chargax_actions)
            obs = timestep.observation

            step_log.append(StepRecord(
                t            = t,
                actions      = {k: float(v) for k, v in actions.items()},
                reward       = float(timestep.reward),
                profit_delta = float(state.profit),
                served       = int(state.served_customers),
                rejected     = int(state.rejected_customers),
                step_revenue = step_revenue,
                step_cost    = step_cost,
            ))

            done = bool(timestep.terminated) or bool(timestep.truncated)

        return ChargaxSimResults.from_final_state(
            controller_name = controller.__class__.__name__,
            seed            = seed,
            state           = state,
            step_log        = step_log,
            departures_soc  = departures_soc,
        )

    def run_benchmark(
        self,
        controllers: dict[str, BaseController],
        seeds: list[int],
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        for name, ctrl in controllers.items():
            for seed in seeds:
                result = self.run_episode(ctrl, seed)
                storage.save(result, output_dir / name / f"seed_{seed}.json")
