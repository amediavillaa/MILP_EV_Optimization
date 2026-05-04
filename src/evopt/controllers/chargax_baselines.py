from __future__ import annotations

import numpy as np

from evopt.controllers.base_controller import BaseController


class MaxChargeController(BaseController):
    """Always charges every connected vehicle at its port's maximum current."""

    def compute_action(self, state: dict) -> dict[int, float]:
        return {
            port_j: state["I_max"][port_j]
            for port_j in state["assignments"].values()
        }


class RandomController(BaseController):
    """Charges each vehicle at a uniformly random current in [0, I_max]."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def compute_action(self, state: dict) -> dict[int, float]:
        return {
            port_j: float(self._rng.uniform(0.0, state["I_max"][port_j]))
            for port_j in state["assignments"].values()
        }
