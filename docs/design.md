# System Design

## Objective
This repository implements and evaluates EV charging control methods under site power constraints and dynamic electricity prices.

## Core components
- Environment wrapper: converts simulator state into controller inputs
- Controllers: baseline and optimization-based charging policies
- Optimization layer: variables, constraints, objective, solver
- Experiment runner: executes simulation scenarios
- Metrics: computes profit, demand satisfaction, and runtime summaries

## Design principles
- Keep controller logic separate from optimization model construction
- Keep metrics separate from experiment orchestration
- Use configuration files instead of hardcoding experiment settings
- Ensure code can support both simple baselines and more advanced LP extensions

## Initial controllers
- Equal-share baseline
- LP-based controller

## Planned extensions
- Receding-horizon control
- Battery storage
- Adaptive departure assumptions
- Trade-off analysis between profit and customer satisfaction