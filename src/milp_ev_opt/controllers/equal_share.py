from milp_ev_opt.controllers.base_controller import BaseController


class EqualShareController(BaseController):
    """Splits available grid capacity equally across all present vehicles."""

    def compute_action(self, state: dict) -> dict[int, float]:
        present_cars = state.get("present_cars", {})
        assignments  = state.get("assignments", {})

        if not present_cars:
            return {}

        N       = len(present_cars)
        P_share = state["P_max"] / N       # Watts per car

        actions = {}
        for car_id, port_j in assignments.items():
            V      = state["V"][port_j]
            I_max  = state["I_max"][port_j]
            car    = present_cars[car_id]

            # Cap by port hardware limit
            I_port = P_share / V
            I_capped = min(I_port, I_max)

            # Cap by energy still needed
            energy_needed_kwh = max(0.0, car["s_target"] - car["soc_now"])
            I_needed = energy_needed_kwh / (state["delta_t"] * V / 1000.0)

            actions[port_j] = min(I_capped, I_needed)

        return actions