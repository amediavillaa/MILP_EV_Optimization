from __future__ import annotations

from pyomo.environ import value

from evopt.controllers.base_controller import BaseController
from evopt.optimization.model import build_rolling_model
from evopt.optimization.solver import solve


class LPController(BaseController):
    """
    Rolling MPC controller: re-solves the LP at every timestep using the
    latest observed state. Jointly optimises EV charging and BESS arbitrage
    over a `horizon_steps`-step lookahead window in a single LP solve.
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
        must_serve: bool = True,
    ) -> None:
        self.horizon_steps    = horizon_steps
        self.solver           = solver
        self.v_bess           = v_bess
        self.I_high           = I_high
        self.I_low            = I_low
        self.socb_min         = socb_min
        self.socb_max         = socb_max
        self.minutes_per_step = minutes_per_step
        self.must_serve       = must_serve

    def reset(self) -> None:
        pass

    def compute_action(self, state: dict) -> dict[int, float]:
        t = state["t"]

        if not state["present_cars"]:
            return {}

        plan = self._solve(state)
        return plan.get(t, {})

    def _build_lp_data(self, state: dict) -> dict:
        T_max = 24 * 60 // self.minutes_per_step - 1   # assumes single-day planning epoch

        t           = state["t"]
        J           = state["J"]
        assignments = state["assignments"]
        # Only consider cars that have a port assignment; unassigned cars are
        # excluded so P_car_max can be computed safely. build_rolling_model
        # returns None when the resulting car list is empty.
        cars        = [cid for cid in state["present_cars"] if cid in assignments]

        t_end  = min(t + self.horizon_steps - 1, T_max)
        window = range(t, t_end + 1)

        return {
            "J":        J,
            "T":        T_max,
            "delta_t":  state["delta_t"],
            "P_max":    state["P_max"],
            "V":        {**state["V"], J + 1: self.v_bess},
            "I_max":    state["I_max"],
            "I_high":   state.get("I_high", self.I_high),
            "I_low":    state.get("I_low",  self.I_low),
            "p_buy":       state["p_buy"],
            "p_sell":      state["p_sell"],
            # Background load L: Chargax does not expose a forecast, so we
            # conservatively assume zero.  This makes the grid-cap constraint
            # slightly optimistic; the wrapper's to_chargax_actions scales down
            # any actions that exceed P_max at execution time.
            "L":        {step: 0.0 for step in window},
            "assignments": assignments,
            "dep":      {cid: c["t_max"]    for cid, c in state["present_cars"].items()},
            "s_cap":    {cid: c["s_cap"]    for cid, c in state["present_cars"].items()},
            "s_target": {cid: c["s_target"] for cid, c in state["present_cars"].items()},
            "s_min":    {cid: 0.0 for cid in cars},
            "P_car_max": {
                cid: state["V"][assignments[cid]] * state["I_max"][assignments[cid]]
                for cid in cars
            },
            "r_car":     {(cid, step): 1.0 for cid in cars for step in window},
            "SoCB_min":  self.socb_min,
            "SoCB_max":  self.socb_max,
            "r_bess_ch": {step: 1.0 for step in window},
            "r_bess_dis":{step: 1.0 for step in window},
        }

    def _solve(self, state: dict) -> dict[int, dict[int, float]]:
        t           = state["t"]
        J           = state["J"]
        assignments = state["assignments"]
        socb_now    = state.get("socb_now", 0.0)
        data        = self._build_lp_data(state)
        # Clamp soc_now to s_target: Chargax discretisation can overshoot the
        # LP's planned energy, making soc_now > s_target.  The rolling model's
        # C8 constraint (soc_car <= s_target) is then immediately infeasible,
        # which silences charging for every car at that step.
        soc_now = {
            cid: min(c["soc_now"], data["s_target"][cid])
            for cid, c in state["present_cars"].items()
        }

        bare = not self.must_serve
        m = build_rolling_model(data, t, self.horizon_steps, assignments, soc_now, socb_now,
                                bare=bare)
        if m is None:
            return {}

        try:
            solve(m, solver=self.solver)
        except Exception:
            if not bare:
                # Must-serve + grid cap can be jointly infeasible in rare tight scenarios;
                # rebuild without must-serve as a last resort so the step is never skipped.
                try:
                    m = build_rolling_model(data, t, self.horizon_steps, assignments,
                                            soc_now, socb_now, bare=True)
                    if m is None:
                        return {}
                    solve(m, solver=self.solver)
                except Exception:
                    return {}
            else:
                return {}

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
