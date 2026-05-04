from __future__ import annotations

import jax.numpy as jnp

from evopt.env.action_mapper import discretize_amps


class ChargaxWrapper:
    """
    Translates between Chargax observations and the data-dict format
    that BaseController.compute_action expects.

    Port indexing convention:
        Chargax uses 0-indexed charger slots  (j = 0, 1, ..., J-1).
        The MILP and controllers use 1-indexed port IDs (j = 1, 2, ..., J).
        This wrapper converts between the two: port_j = charger_j + 1.
    """

    def __init__(
        self,
        n_ports: int,
        v: float,
        i_max: float,
        p_max_kw: float,
        num_discretization_levels: int = 10,
        minutes_per_step: int = 5,
    ) -> None:
        self.J                        = n_ports
        self.V    = {j + 1: v     for j in range(n_ports)}
        self.I_max = {j + 1: i_max for j in range(n_ports)}
        self.P_max_w                  = p_max_kw * 1000.0
        self.num_discretization_levels = num_discretization_levels
        self.minutes_per_step         = minutes_per_step

        self._charger_to_car: dict[int, int]   = {}
        self._prev_connected: set[int]          = set()
        self._prev_soc: dict[int, float]        = {}
        self._next_car_id: int                  = 0

    def reset(self) -> None:
        self._charger_to_car = {}
        self._prev_connected = set()
        self._prev_soc       = {}
        self._next_car_id    = 0

    def extract_state(self, obs: dict, chargax_state) -> dict:
        evse = obs["evses"]
        t    = int(chargax_state.timestep)

        now_connected = {
            j for j in range(self.J)
            if bool(evse.charger_is_car_connected[j])
        }
        departures   = self._prev_connected - now_connected
        new_arrivals = now_connected - self._prev_connected

        # Always process departures before arrivals (prevents same-step ID reuse).
        # Use the SOC captured at the previous step, since the current obs for a
        # departed charger reflects post-disconnect (stale/zero) values.
        departed_socs: list[float] = []
        for j in sorted(departures):
            departed_socs.append(self._prev_soc.get(j, float(evse.car_battery_now_kw[j])))
            del self._charger_to_car[j]
            self._prev_soc.pop(j, None)

        for j in sorted(new_arrivals):
            self._charger_to_car[j] = self._next_car_id
            self._next_car_id += 1

        self._prev_connected = now_connected

        present_cars: dict = {}
        assignments:  dict = {}
        for j, car_id in self._charger_to_car.items():
            soc_now    = float(evse.car_battery_now_kw[j])
            s_cap      = float(evse.car_battery_capacity_kw[j])
            desired_pct = float(evse.car_desired_battery_percentage[j])
            s_target   = desired_pct * s_cap
            time_left_min = float(evse.car_time_till_leave[j])
            t_max      = t + max(1, round(time_left_min / self.minutes_per_step))

            present_cars[car_id] = {
                "soc_now":  soc_now,
                "s_target": s_target,
                "s_cap":    s_cap,
                "t_max":    t_max,
            }
            assignments[car_id] = j + 1   # 0-indexed → 1-indexed port
            self._prev_soc[j]   = soc_now  # snapshot for next departure lookup

        steps_per_hour = 60 // self.minutes_per_step
        future_buy  = [float(p) for p in obs["future_buy_prices"]]
        future_sell = [float(p) for p in obs["future_sell_prices"]]

        p_buy: dict[int, float] = {}
        p_sell: dict[int, float] = {}
        for h, (bp, sp) in enumerate(zip(future_buy, future_sell)):
            for s in range(steps_per_hour):
                step = t + h * steps_per_hour + s
                p_buy[step]  = bp
                p_sell[step] = sp

        return {
            "t":            t,
            "delta_t":      self.minutes_per_step / 60.0,
            "J":            self.J,
            "P_max":        self.P_max_w,
            "V":            self.V,
            "I_max":        self.I_max,
            "p_buy":        p_buy,
            "p_sell":       p_sell,
            "present_cars": present_cars,
            "assignments":  assignments,
            "departed_socs": departed_socs,
        }

    def to_chargax_actions(self, actions: dict[int, float]) -> dict:
        """Convert {port_j (1-indexed): amps} to Chargax MultiDiscrete action dict."""
        levels = []
        for j in range(self.J):
            port_j = j + 1
            amps   = actions.get(port_j, 0.0)
            levels.append(discretize_amps(amps, self.I_max[port_j],
                                          self.num_discretization_levels))
        return {
            "evses":     jnp.array(levels, dtype=jnp.int32),
            "batteries": jnp.array([0],    dtype=jnp.int32),
        }
