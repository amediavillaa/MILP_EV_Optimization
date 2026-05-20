import math


def discretize_amps(amps: float, i_max: float, num_levels: int,
                    bidirectional: bool = False) -> int:
    """Convert a continuous current (A) to a discrete Chargax charging level.

    Unidirectional (bidirectional=False):
        Action space [0, N].  0 = off, N = max charge.
        level = ceil(amps / i_max * N), clamped to [0, N].

    Bidirectional (bidirectional=True):
        Action space [0, 2N].  0 = max discharge, N = idle, 2N = max charge.
        Chargax formula: desired_output = (level/N - 1) * P_max
        For EV ports we only charge (amps >= 0), so output is in [N, 2N].
        level = N + ceil(amps / i_max * N), clamped to [N, 2N].

    Uses ceiling rounding so LP plans execute at-or-above the requested level.
    Currents below half a discretisation step are treated as zero/idle to avoid
    spurious non-zero levels from numerical noise.
    """
    if i_max <= 0:
        return num_levels if bidirectional else 0
    normalised = amps / i_max * num_levels
    if bidirectional:
        if normalised < 0.5:
            return num_levels  # idle
        return min(2 * num_levels, num_levels + math.ceil(normalised - 1e-9))
    else:
        if normalised < 0.5:
            return 0
        return min(num_levels, math.ceil(normalised - 1e-9))
