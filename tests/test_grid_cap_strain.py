import pytest


# ── Validation ────────────────────────────────────────────────────────────────

def test_validation_rejects_zero():
    from evopt.analysis.utils import validate_grid_cap_strain
    with pytest.raises(ValueError, match="must be in"):
        validate_grid_cap_strain([0.0])


def test_validation_rejects_above_one():
    from evopt.analysis.utils import validate_grid_cap_strain
    with pytest.raises(ValueError, match="must be in"):
        validate_grid_cap_strain([1.1])


def test_validation_accepts_valid():
    from evopt.analysis.utils import validate_grid_cap_strain
    validate_grid_cap_strain([1.0])
    validate_grid_cap_strain([0.5])
    validate_grid_cap_strain([1.0, 0.75, 0.5, 0.25])


# ── Multiplier property ────────────────────────────────────────────────────────

def test_multiplier_applied_to_wrapper():
    from evopt.env.chargax_wrapper import ChargaxWrapper
    wrapper = ChargaxWrapper(n_ports=3, v=400.0, i_max=32.0, p_max_kw=9.0)
    assert wrapper.P_max_w == pytest.approx(9000.0)


# ── EqualShare integration ─────────────────────────────────────────────────────

def _make_state(p_max_kw: float) -> dict:
    P_max_w = p_max_kw * 1000.0
    return {
        "t": 0,
        "delta_t": 5 / 60,
        "J": 3,
        "P_max": P_max_w,
        "V":     {1: 400.0, 2: 400.0, 3: 400.0},
        "I_max": {1: 32.0,  2: 32.0,  3: 32.0},
        "p_buy":  {0: 0.10},
        "p_sell": {0: 0.75},
        "present_cars": {
            0: {"soc_now": 5.0, "s_target": 20.0, "s_cap": 60.0, "t_max": 20},
            1: {"soc_now": 5.0, "s_target": 20.0, "s_cap": 60.0, "t_max": 20},
            2: {"soc_now": 5.0, "s_target": 20.0, "s_cap": 60.0, "t_max": 20},
        },
        "assignments": {0: 1, 1: 2, 2: 3},
        "departed_fulfillments": [],
    }


def test_strain_save_path():
    from pathlib import Path
    from evopt.analysis.utils import strain_save_path

    result = strain_save_path("results/bench.csv", 0.75)
    assert result == Path("results/bench_s0_75/bench.csv")

    result_one = strain_save_path("results/bench.csv", 1.0)
    assert result_one == Path("results/bench_s1_00/bench.csv")


def test_format_experiment_id_with_strain():
    from evopt.analysis.utils import format_experiment_id

    base = format_experiment_id(ports=[3], horizons=[12], tariff="0.75")
    assert "_s" not in base

    with_strain = format_experiment_id(ports=[3], horizons=[12], tariff="0.75", grid_cap_strain=0.75)
    assert with_strain == base + "_s0_75"


def test_equal_share_respects_tighter_cap():
    from evopt.controllers.equal_share import EqualShareController
    ctrl = EqualShareController()

    state_full = _make_state(18.0)   # strain = 1.0  (3 ports × 6 kW)
    state_half = _make_state(9.0)    # strain = 0.5

    actions_full = ctrl.compute_action(state_full)
    actions_half = ctrl.compute_action(state_half)

    V = 400.0
    power_full = sum(a * V for a in actions_full.values())
    power_half = sum(a * V for a in actions_half.values())

    # EqualShare splits P_max/N per car; halving P_max must halve total power
    assert power_full > 0, "full-cap state produced zero total power — check _make_state"
    assert power_half <= power_full * 0.5 + 1e-6
