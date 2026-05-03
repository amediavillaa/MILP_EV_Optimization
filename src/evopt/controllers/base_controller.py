class BaseController:
    """Abstract interface for all charging controllers."""

    def compute_action(self, state: dict) -> dict[int, float]:
        """Return {port_j: amps} for the current state dict."""
        raise NotImplementedError("Subclasses must implement compute_action().")

    def reset(self) -> None:
        """Called at the start of each episode. Stateless controllers may leave this as a no-op."""
        pass