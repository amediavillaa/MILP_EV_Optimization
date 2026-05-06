def discretize_amps(amps: float, i_max: float, num_levels: int) -> int:
    """Convert a continuous current (A) to a discrete charging level [0, num_levels]."""
    if i_max <= 0:
        return 0
    level = round(amps / i_max * num_levels)
    return max(0, min(num_levels, level))
