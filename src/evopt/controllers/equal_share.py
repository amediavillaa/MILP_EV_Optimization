from evopt.controllers.base_controller import BaseController


class EqualShareController(BaseController):
    """Simple baseline that splits available capacity equally across active vehicles."""

    def compute_action(self, state):
        active_vehicles = state.get("active_vehicles", [])
        capacity = state.get("site_capacity_kw", 0.0)

        if not active_vehicles:
            return {}

        share = capacity / len(active_vehicles)
        return {vehicle_id: share for vehicle_id in active_vehicles}