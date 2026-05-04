from __future__ import annotations

from pyomo.environ import value

from evopt.controllers.base_controller import BaseController
from evopt.optimization.model import build_rolling_model
from evopt.optimization.solver import solve


class LPController(BaseController):
    """
    Stateful rolling MPC controller.

    Re-solves the MILP every `horizon_steps` steps and holds the resulting
    plan for intermediate steps. At each solve, the model looks ahead
    `horizon_steps` time steps (e.g. 12 × 5 min = 1 hour).
    """

    def __init__(
        self,
        horizon_steps: int = 12,
        solver: str = "highs",
        M_big: float = 100.0,
        epsilon: float = 0.1,
        minutes_per_step: int = 5,
    ) -> None:
        self.horizon_steps    = horizon_steps
        self.solver           = solver
        self.M_big            = M_big
        self.epsilon          = epsilon
        self.minutes_per_step = minutes_per_step

        self._plan: dict[int, dict[int, float]] | None = None

    def reset(self) -> None:
        self._plan = None

    def compute_action(self, state: dict) -> dict[int, float]:
        t           = state["t"]
        present     = state["present_cars"]
        assignments = state["assignments"]

        if not present:
            return {}

        if self._plan is None or t % self.horizon_steps == 0:
            self._plan = self._solve(state)

        return self._plan.get(t, {})

    def _build_milp_data(self, state: dict) -> dict:
        minutes_per_day = 24 * 60
        T_max  = minutes_per_day // self.minutes_per_step - 1   # 287 for 5-min

        cars = state["present_cars"]
        return {
            "J":        state["J"],
            "T":        T_max,
            "delta_t":  state["delta_t"],
            "P_max":    state["P_max"],
            "M_big":    self.M_big,
            "epsilon":  self.epsilon,
            "V":        state["V"],
            "I_max":    state["I_max"],
            "p_buy":    state["p_buy"],
            "p_sell":   state["p_sell"],
            "s_target": {cid: c["s_target"] for cid, c in cars.items()},
            "s_cap":    {cid: c["s_cap"]    for cid, c in cars.items()},
        }

    def _solve(self, state: dict) -> dict[int, dict[int, float]]:
        t           = state["t"]
        assignments = state["assignments"]
        soc_now     = {cid: c["soc_now"] for cid, c in state["present_cars"].items()}
        data        = self._build_milp_data(state)

        m = build_rolling_model(data, t, self.horizon_steps, assignments, soc_now)
        if m is None:
            return {}

        solve(m, solver=self.solver)

        t_end = min(t + self.horizon_steps - 1, data["T"])
        plan: dict[int, dict[int, float]] = {}
        for step in range(t, t_end + 1):
            step_actions: dict[int, float] = {}
            for j in m.J:
                step_actions[int(j)] = float(value(m.I_charge[j, step]))
            plan[step] = step_actions
        return plan
