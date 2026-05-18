from __future__ import annotations

import jax.numpy as jnp

from evopt.env.action_mapper import discretize_amps


class ChargaxWrapper:
    """
    Translates between Chargax observations and the data-dict format
    that BaseController.compute_action expects.

    Port indexing convention:
        Chargax uses 0-indexed charger slots  (j = 0, 1, ..., J-1).
        The LP uses 1-indexed port IDs (j = 1, 2, ..., J).
        BESS occupies port J+1 in the LP; it has no Chargax charger slot.
    """

    def __init__(
        self,
        n_ports: int,
        v: float,
        i_max: float,
        p_max_kw: float,
        num_discretization_levels: int = 10,
        minutes_per_step: int = 5,
        # EV customer tariff: fixed float (€/kWh) or None to use live Chargax p_sell
        ev_tariff: float | None = 0.75,
        # BESS params (optional — if None, BESS is ignored)
        v_bess: float | None = None,
        I_high: float | None = None,   # max BESS charging current (A)
        I_low: float | None = None,    # max BESS discharging current (A)
        socb_min: float = 0.0,
        socb_max: float | None = None,
        p_bess_max_kw: float | None = None,
        allow_discharging: bool = False,       # EVSE bidirectionality (V2G)
        allow_bess_discharging: bool = False,  # BESS discharge in LP
    ) -> None:
        self.J                         = n_ports
        self.V                         = {j + 1: v     for j in range(n_ports)}
        self.I_max                     = {j + 1: i_max for j in range(n_ports)}
        self.P_max_w                   = p_max_kw * 1000.0
        self.num_discretization_levels = num_discretization_levels
        self.minutes_per_step          = minutes_per_step
        self.ev_tariff                 = ev_tariff

        self.v_bess            = v_bess
        self.I_high            = I_high
        self.I_low             = I_low
        self.socb_min          = socb_min
        self.socb_max          = socb_max
        self.p_bess_max_kw     = p_bess_max_kw
        self.allow_discharging      = allow_discharging
        self.allow_bess_discharging = allow_bess_discharging

        self._charger_to_car: dict[int, int] = {}
        self._prev_connected: set[int]        = set()
        self._prev_soc: dict[int, float]      = {}
        self._next_car_id: int                = 0

        if self.v_bess is not None:
            if self.I_high is None or self.I_low is None or self.p_bess_max_kw is None or self.socb_max is None:
                raise ValueError(
                    "When v_bess is set, I_high, I_low, p_bess_max_kw, and socb_max "
                    "must all be provided."
                )

    def reset(self) -> None:
        self._charger_to_car = {}
        self._prev_connected = set()
        self._prev_soc       = {}
        self._next_car_id    = 0

    def extract_state(self, obs: dict, chargax_state) -> dict:
        evse_raw = obs["evses"]
        evse = evse_raw[0] if isinstance(evse_raw, list) else evse_raw
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
            soc_now       = float(evse.car_battery_now_kw[j])
            s_cap         = float(evse.car_battery_capacity_kw[j])
            desired_pct   = float(evse.car_desired_battery_percentage[j])
            s_target      = desired_pct * s_cap
            time_left_min = float(evse.car_time_till_leave[j])
            t_max         = t + max(1, round(time_left_min / self.minutes_per_step))

            present_cars[car_id] = {
                "soc_now":  soc_now,
                "s_target": s_target,
                "s_cap":    s_cap,
                "t_max":    t_max,
            }
            assignments[car_id] = j + 1
            self._prev_soc[j]   = soc_now  # snapshot for next departure lookup

        steps_per_hour = 60 // self.minutes_per_step
        future_buy  = [float(p) for p in obs["future_buy_prices"]]
        future_sell = [float(p) for p in obs["future_sell_prices"]]

        # p_buy  = grid electricity buy price (market, varies by hour)
        # p_sell = EV customer tariff: fixed (self.ev_tariff) or dynamic market price
        p_buy: dict[int, float]  = {}
        p_sell: dict[int, float] = {}
        for h, (bp, sp) in enumerate(zip(future_buy, future_sell)):
            for s in range(steps_per_hour):
                step = t + h * steps_per_hour + s
                p_buy[step]  = bp
                p_sell[step] = self.ev_tariff if self.ev_tariff is not None else sp

        state: dict = {
            "t":             t,
            "delta_t":       self.minutes_per_step / 60.0,
            "J":             self.J,
            "P_max":         self.P_max_w,
            "V":             self.V,
            "I_max":         self.I_max,
            "p_buy":         p_buy,
            "p_sell":        p_sell,
            "present_cars":  present_cars,
            "assignments":   assignments,
            "departed_socs": departed_socs,
        }

        if self.v_bess is not None:
            if not obs.get("batteries"):
                raise RuntimeError(
                    "ChargaxWrapper is configured for BESS (v_bess is set) but the "
                    "Chargax observation contains no batteries. Ensure the station "
                    "was built with build_station_with_battery."
                )
            batt = obs["batteries"][0]
            state["socb_now"] = float(batt.battery_now)
            state["socb_min"] = self.socb_min
            state["socb_max"] = self.socb_max
            state["I_high"]   = self.I_high
            state["I_low"]    = self.I_low if self.allow_bess_discharging else 0.0

        return state

    def to_chargax_actions(self, actions: dict[int, float]) -> dict:
        """Convert {port_j (1-indexed): amps} to Chargax MultiDiscrete action dict.

        Port J+1 is the BESS. LP convention: positive bess_net_amps = discharging.
        Chargax convention: battery level is an integer in [0, 2*N] where N =
        num_discretization_levels; level N = idle, level 0 = max discharge, level 2*N
        = max charge.  Chargax internally computes:
            desired_output_kw = (level / N - 1) * max_throughput_kw
        so charging requires level > N and discharging requires level < N.

        Mapping from LP power (positive = discharge) to Chargax level:
            level = round((1 - bess_power_kw / p_bess_max_kw) * N)
        clamped to [0, 2*N].

        Hard grid cap: total EV grid draw is clipped to P_max before discretisation
        so that all controllers, including baselines, respect the same physical limit.
        BESS discharging reduces grid draw; BESS charging adds to it.
        """
        p_max_kw = self.P_max_w / 1000.0

        ev_kw = sum(
            actions.get(j + 1, 0.0) * self.V[j + 1] / 1000.0
            for j in range(self.J)
        )
        bess_net_kw = 0.0
        if self.v_bess is not None:
            bess_net_kw = actions.get(self.J + 1, 0.0) * self.v_bess / 1000.0  # positive = discharging

        grid_draw_kw = ev_kw - bess_net_kw  # net draw from grid
        scale = (p_max_kw / grid_draw_kw) if grid_draw_kw > p_max_kw else 1.0

        levels = []
        for j in range(self.J):
            port_j = j + 1
            amps   = actions.get(port_j, 0.0) * scale
            levels.append(discretize_amps(amps, self.I_max[port_j],
                                          self.num_discretization_levels,
                                          bidirectional=self.allow_discharging))

        if self.v_bess is not None:
            bess_net_amps = actions.get(self.J + 1, 0.0)
            bess_power_kw = bess_net_amps * self.v_bess / 1000.0   # positive = discharge
            level = round((1 - bess_power_kw / self.p_bess_max_kw) * self.num_discretization_levels)
            level = max(0, min(2 * self.num_discretization_levels, level))
            batteries = [jnp.array(level, dtype=jnp.int32)]
        else:
            batteries = []

        return {
            "evses":     [jnp.array(levels, dtype=jnp.int32)],
            "batteries": batteries,
        }
