from __future__ import annotations

from pyomo.environ import value

from evopt.controllers.base_controller import BaseController
from evopt.optimization.model import build_rolling_model
from evopt.optimization.solver import solve


class LPController(BaseController):
    """
    Stateful rolling MPC controller.

    Re-solves the LP at t=0 and every `horizon_steps` thereafter, caching
    the plan for intermediate steps. Jointly optimises EV charging and BESS
    arbitrage in a single LP solve.
    """

    def __init__(
        self,
        horizon_steps: int = 12,
        solver: str = "highs",
        v_bess: float = 400.0,
        I_high: float = 25.0,
        I_low: float = 25.0,
        socb_min: float = 3.0,
        socb_max: float = 30.0,
        minutes_per_step: int = 5,
    ) -> None:
        self.horizon_steps    = horizon_steps
        self.solver           = solver
        self.v_bess           = v_bess
        self.I_high           = I_high
        self.I_low            = I_low
        self.socb_min         = socb_min
        self.socb_max         = socb_max
        self.minutes_per_step = minutes_per_step

        self._plan: dict[int, dict[int, float]] | None = None

    def reset(self) -> None:
        self._plan = None

    def compute_action(self, state: dict) -> dict[int, float]:
        t = state["t"]

        if not state["present_cars"]:
            return {}

        if self._plan is None or t % self.horizon_steps == 0:
            self._plan = self._solve(state)

        return self._plan.get(t, {})

    def _build_lp_data(self, state: dict) -> dict:
        T_max = 24 * 60 // self.minutes_per_step - 1   # 287 for 5-min steps

        t           = state["t"]
        J           = state["J"]
        cars        = list(state["present_cars"].keys())
        assignments = state["assignments"]

        t_end  = min(t + self.horizon_steps - 1, T_max)
        window = range(t, t_end + 1)

        return {
            "J":        J,
            "T":        T_max,
            "delta_t":  state["delta_t"],
            "P_max":    state["P_max"],
            "V":        {**state["V"], J + 1: self.v_bess},
            "I_max":    state["I_max"],
            "I_high":   self.I_high,
            "I_low":    self.I_low,
            "p_buy":    state["p_buy"],
            "p_sell":   state["p_sell"],
            "L":        {step: 0.0 for step in window},
            "assignments": assignments,
            "dep":      {cid: c["t_max"] for cid, c in state["present_cars"].items()},
            "s_cap":    {cid: c["s_cap"] for cid, c in state["present_cars"].items()},
            "s_min":    {cid: 0.0 for cid in cars},
            "P_car_max": {
                cid: state["V"][assignments[cid]] * state["I_max"][assignments[cid]]
                for cid in cars
            },
            "r_car":     {(cid, step): 1.0 for cid in cars for step in window},
            "SoCB_init": state["socb_now"],
            "SoCB_min":  self.socb_min,
            "SoCB_max":  self.socb_max,
            "r_bess_ch": {step: 1.0 for step in window},
            "r_bess_dis":{step: 1.0 for step in window},
        }

    def _solve(self, state: dict) -> dict[int, dict[int, float]]:
        t           = state["t"]
        J           = state["J"]
        assignments = state["assignments"]
        soc_now     = {cid: c["soc_now"] for cid, c in state["present_cars"].items()}
        socb_now    = state["socb_now"]
        data        = self._build_lp_data(state)

        m = build_rolling_model(data, t, self.horizon_steps, assignments, soc_now, socb_now)
        if m is None:
            return {}

        solve(m, solver=self.solver)

        T_max = data["T"]
        t_end = min(t + self.horizon_steps - 1, T_max)

        plan: dict[int, dict[int, float]] = {}
        for step in range(t, t_end + 1):
            actions: dict[int, float] = {}
            for j in m.J_ev:
                actions[int(j)] = float(value(m.I_ev[j, step]))
            bess_net = float(value(m.I_bess_dis[step])) - float(value(m.I_bess_ch[step]))
            actions[J + 1] = bess_net   # positive = discharging (LP convention)
            plan[step] = actions

        return plan
