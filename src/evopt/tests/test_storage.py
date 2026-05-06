import json
from pathlib import Path
import pytest
from evopt.benchmarking.results import ChargaxSimResults, StepRecord
from evopt.benchmarking import storage


def _make_result(name: str, seed: int, profit: float) -> ChargaxSimResults:
    return ChargaxSimResults(
        controller_name=name,
        seed=seed,
        episode_date="2023-01-01",
        net_profit=profit,
        total_revenue=profit + 2.0,
        total_cost=2.0,
        served_customers=5,
        rejected_customers=1,
        mean_soc_at_departure=0.9,
        step_log=[StepRecord(t=0, actions={1: 16.0}, reward=0.1,
                             profit_delta=0.05, served=0, rejected=0)],
    )


def test_save_and_load_roundtrip(tmp_path):
    result = _make_result("milp_h12", 0, 10.5)
    path = tmp_path / "milp_h12" / "seed_0.json"
    storage.save(result, path)
    loaded = storage.load(path)
    assert loaded.controller_name == "milp_h12"
    assert loaded.net_profit == pytest.approx(10.5)
    assert loaded.seed == 0
    assert len(loaded.step_log) == 1
    assert loaded.step_log[0].t == 0


def test_save_creates_parent_dirs(tmp_path):
    result = _make_result("equal_share", 1, 8.0)
    path = tmp_path / "deep" / "nested" / "seed_1.json"
    storage.save(result, path)
    assert path.exists()


def test_build_summary_returns_dataframe(tmp_path):
    for name, profit in [("milp_h12", 10.0), ("equal_share", 7.0)]:
        for seed in range(2):
            r = _make_result(name, seed, profit + seed * 0.1)
            storage.save(r, tmp_path / name / f"seed_{seed}.json")

    df = storage.build_summary(tmp_path)
    assert set(df["controller"].unique()) == {"milp_h12", "equal_share"}
    assert "net_profit" in df.columns
    assert "gap_to_best" in df.columns


def test_build_summary_empty_dir(tmp_path):
    df = storage.build_summary(tmp_path)
    assert len(df) == 0
