from milp_ev_opt.experiments.station_configs import build_simple_station
from chargax import ChargingStation


def test_build_simple_station_returns_charging_station():
    station = build_simple_station(n_ports=3, v=400.0, i_max=32.0, p_max_kw=20.0)
    assert isinstance(station, ChargingStation)


import jax
import jax.numpy as jnp
from milp_ev_opt.benchmarking.runner import BenchmarkRunner
from milp_ev_opt.benchmarking import storage
from milp_ev_opt.env.chargax_wrapper import ChargaxWrapper
from milp_ev_opt.controllers.equal_share import EqualShareController
from chargax import Chargax


def _make_env_and_wrapper():
    station = build_simple_station(n_ports=2, v=400.0, i_max=32.0, p_max_kw=10.0)
    env = Chargax(
        station=station,
        minutes_per_timestep=5,
        allow_discharging=False,
        renormalize_currents=False,
    )
    wrapper = ChargaxWrapper(
        n_ports=2, v=400.0, i_max=32.0, p_max_kw=10.0,
        num_discretization_levels=10, minutes_per_step=5,
    )
    return env, wrapper


def test_benchmark_runner_completes_one_episode():
    env, wrapper = _make_env_and_wrapper()
    runner = BenchmarkRunner(env, wrapper)
    result = runner.run_episode(EqualShareController(), seed=0)

    assert result.controller_name == "EqualShareController"
    assert result.seed == 0
    assert isinstance(result.net_profit, float)
    assert len(result.step_log) == 288   # 24h * 12 steps/h


def test_benchmark_runner_saves_results(tmp_path):
    env, wrapper = _make_env_and_wrapper()
    runner = BenchmarkRunner(env, wrapper)
    runner.run_benchmark(
        controllers={"equal_share": EqualShareController()},
        seeds=[0, 1],
        output_dir=tmp_path,
    )
    files = list(tmp_path.glob("**/*.json"))
    assert len(files) == 2
