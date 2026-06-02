from __future__ import annotations

from evopt.controllers.base_controller import BaseController
from evopt.optimization.model import build_ev_lp_model
from evopt.optimization.solver import solve
from pyomo.environ import value


class ScenarioCollector:
    """Records every state snapshot from a collection-pass episode.

    Call `record(state)` at every step with the output of
    ChargaxWrapper.extract_state. Then call `build_scenario()` to get
    the full-episode data dict required by build_ev_lp_model.
    """

    def __init__(
        self,
        i_high:   float = 25.0,
        i_low:    float = 25.0,
        socb_min: float = 3.0,
        socb_max: float = 30.0,
    ) -> None:
        self._i_high   = i_high
        self._i_low    = i_low
        self._socb_min = socb_min
        self._socb_max = socb_max
        self._first_seen:    dict[int, dict]  = {}   # car_id -> {arr, s_init, port_j, s_cap, s_target}
        self._last_t_max:    dict[int, int]   = {}   # car_id -> most recent t_max estimate
        self._p_buy:         dict[int, float] = {}
        self._p_sell:        dict[int, float] = {}
        self._scenario_meta: dict             = {}   # J, P_max, V, I_max, delta_t
        self._socb_init:     float | None     = None

    def record(self, state: dict) -> None:
        t = state["t"]

        # Capture station-level metadata from first step
        if not self._scenario_meta:
            self._scenario_meta = {
                "J":       state["J"],
                "P_max":   state["P_max"],
                "V":       dict(state["V"]),
                "I_max":   dict(state["I_max"]),
                "delta_t": state["delta_t"],
            }

        if self._socb_init is None:
            self._socb_init = state.get("socb_now", 0.0)

        # Accumulate prices (later steps overwrite earlier for the same key,
        # but since future_buy_prices extends forward, union covers all steps)
        self._p_buy.update(state["p_buy"])
        self._p_sell.update(state["p_sell"])

        present = state.get("present_cars", {})
        assignments = state.get("assignments", {})
        for car_id, car in present.items():
            if car_id not in self._first_seen:
                self._first_seen[car_id] = {
                    "arr":    t,
                    "s_init": car["soc_now"],
                    "port_j": assignments.get(car_id, 1),
                    "s_cap":  car["s_cap"],
                    "s_target": car["s_target"],
                }
            self._last_t_max[car_id] = car["t_max"]

    def build_scenario(self) -> dict:
        """Return data dict compatible with build_ev_lp_model.

        ChargaxWrapper uses 0-based timestep indices (t=0 at episode start),
        but build_ev_lp_model expects 1-based indices via RangeSet(1, T).
        All time indices are therefore shifted by +1 here so that the offline
        LP model sees t ∈ {1, 2, …, T}.  OfflineLPController.compute_action
        applies the same +1 shift when looking up the schedule.
        """
        if not self._first_seen:
            return {}

        meta = self._scenario_meta
        cars = sorted(self._first_seen.keys())
        # Re-map car IDs to 1-indexed integers for build_ev_lp_model
        id_map = {old: new for new, old in enumerate(cars, start=1)}

        # Shift 0-based time keys to 1-based (build_ev_lp_model uses RangeSet(1, T))
        p_buy_1  = {k + 1: v for k, v in self._p_buy.items()}
        p_sell_1 = {k + 1: v for k, v in self._p_sell.items()}

        T_max = max(p_buy_1.keys()) if p_buy_1 else 288
        I = len(cars)

        # arr / dep are also 0-based raw timesteps; shift to 1-based
        arr         = {id_map[c]: self._first_seen[c]["arr"] + 1      for c in cars}
        dep         = {id_map[c]: self._last_t_max[c] + 1             for c in cars}
        # Clamp dep to T_max so constraints remain feasible
        dep         = {i: min(d, T_max) for i, d in dep.items()}
        s_init      = {id_map[c]: self._first_seen[c]["s_init"]        for c in cars}
        s_target    = {id_map[c]: self._first_seen[c]["s_target"]      for c in cars}
        assignments = {id_map[c]: self._first_seen[c]["port_j"]        for c in cars}

        J      = meta["J"]
        j_bess = J + 1

        # SoC-dependent car charging ratios: flat 1.0 (same as rolling MPC)
        r_car = {
            (id_map[c], t): 1.0
            for c in cars
            for t in range(arr[id_map[c]], dep[id_map[c]] + 1)
        }

        # Use s_target as s_cap in offline LP so it does not over-serve customers
        # (ensures fair comparison with rolling MPC, which stops at s_target)
        s_cap_offline = {id_map[c]: self._first_seen[c]["s_target"] for c in cars}

        P_car_max = {
            id_map[c]: meta["I_max"].get(self._first_seen[c]["port_j"], 32.0)
                       * meta["V"].get(self._first_seen[c]["port_j"], 400.0)
            for c in cars
        }

        return {
            "J":          J,
            "T":          T_max,
            "I":          I,
            "delta_t":    meta["delta_t"],
            "P_max":      meta["P_max"],
            "V":          {**meta["V"], j_bess: meta["V"].get(j_bess, 400.0)},
            "I_max":      meta["I_max"],
            "I_high":     self._i_high,
            "I_low":      self._i_low,
            "p_buy":      p_buy_1,
            "p_sell":     p_sell_1,
            "L":          {t: 0.0 for t in range(1, T_max + 1)},
            "assignments": assignments,
            "arr":        arr,
            "dep":        dep,
            "s_init":     s_init,
            "s_cap":      s_cap_offline,
            "s_min":      {id_map[c]: 0.0 for c in cars},
            "s_target":   s_target,
            "P_car_max":  P_car_max,
            "r_car":      r_car,
            "SoCB_init":  self._socb_init if self._socb_init is not None else 0.0,
            "SoCB_min":   self._socb_min,
            "SoCB_max":   self._socb_max,
            "r_bess_ch":  {t: 1.0 for t in range(1, T_max + 1)},
            "r_bess_dis": {t: 1.0 for t in range(1, T_max + 1)},
        }


class OfflineLPController(BaseController):
    """Replays a pre-computed full-horizon LP schedule step by step.

    Build the schedule dict with build_offline_schedule() before creating
    this controller. The schedule maps {t: {port_j: amps}}.
    """

    def __init__(self, schedule: dict[int, dict[int, float]]) -> None:
        self._schedule = schedule

    def reset(self) -> None:
        pass

    def compute_action(self, state: dict) -> dict[int, float]:
        # state["t"] is 0-based (Chargax convention); schedule keys are 1-based
        return dict(self._schedule.get(state["t"] + 1, {}))


def build_offline_schedule(
    scenario:  dict,
    solver:    str   = "highs",
    eta_bess:  float = 0.95,
) -> dict[int, dict[int, float]]:
    """Solve the full-horizon offline LP and return a schedule dict.

    Returns {t: {port_j: amps}} for t = 1 … T.
    Returns {} if the scenario has no cars or the LP is infeasible.
    """
    if not scenario or scenario.get("I", 0) == 0:
        return {}

    m = build_ev_lp_model(scenario, eta_bess=eta_bess)

    try:
        solve(m, solver=solver)
    except Exception:
        return {}

    J     = scenario["J"]
    T     = scenario["T"]
    j_bess = J + 1

    schedule: dict[int, dict[int, float]] = {}
    for t in range(1, T + 1):
        actions: dict[int, float] = {}
        for j in range(1, J + 1):
            actions[j] = max(0.0, float(value(m.I_ev[j, t])))
        # BESS: positive = discharging (LP convention matches OfflineLPController)
        bess_net = float(value(m.I_bess_dis[t])) - float(value(m.I_bess_ch[t]))
        actions[j_bess] = bess_net
        schedule[t] = actions

    return schedule
