from __future__ import annotations

from evopt.controllers.base_controller import BaseController


class ScenarioCollector:
    """Records every state snapshot from a collection-pass episode.

    Call `record(state)` at every step with the output of
    ChargaxWrapper.extract_state. Then call `build_scenario()` to get
    the full-episode data dict required by build_ev_lp_model.
    """

    def __init__(self) -> None:
        self._first_seen:  dict[int, dict] = {}   # car_id -> {arr, s_init, port_j, s_cap, s_target}
        self._last_t_max:  dict[int, int]  = {}   # car_id -> most recent t_max estimate
        self._p_buy:       dict[int, float] = {}
        self._p_sell:      dict[int, float] = {}
        self._scenario_meta: dict = {}             # J, P_max, V, I_max, delta_t

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
        """Return data dict compatible with build_ev_lp_model."""
        if not self._first_seen:
            return {}

        meta = self._scenario_meta
        cars = sorted(self._first_seen.keys())
        # Re-map car IDs to 1-indexed integers for build_ev_lp_model
        id_map = {old: new for new, old in enumerate(cars, start=1)}

        T_max = max(self._p_buy.keys()) if self._p_buy else 288
        I = len(cars)

        arr         = {id_map[c]: self._first_seen[c]["arr"]     for c in cars}
        dep         = {id_map[c]: self._last_t_max[c]            for c in cars}
        s_init      = {id_map[c]: self._first_seen[c]["s_init"]  for c in cars}
        s_cap       = {id_map[c]: self._first_seen[c]["s_cap"]   for c in cars}
        s_target    = {id_map[c]: self._first_seen[c]["s_target"] for c in cars}
        assignments = {id_map[c]: self._first_seen[c]["port_j"]  for c in cars}

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
            "I_high":     25.0,
            "I_low":      25.0,
            "p_buy":      self._p_buy,
            "p_sell":     self._p_sell,
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
            "SoCB_init":  0.0,
            "SoCB_min":   3.0,
            "SoCB_max":   30.0,
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
        return dict(self._schedule.get(state["t"], {}))
