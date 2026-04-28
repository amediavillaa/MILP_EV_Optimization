class BaseController:
    """Abstract interface for all charging controllers."""

    def compute_action(self, state):
        """Return charging actions for the current environment state."""
        raise NotImplementedError("Subclasses must implement compute_action().")