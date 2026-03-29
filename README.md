# MILP_EV_Optimization
The main objective of this project is to develop a Mixed-Integer Linear Programming (MILP) optimization model that serves as a stable and interpretable baseline for comparison with more advanced approaches, notably Reinforcement Learning (RL), for reducing the operating costs of EV charging stations. 

# EV Charging Optimization with LP and Receding Horizon Control

This repository implements and evaluates a linear programming controller for EV charging power allocation under grid constraints and dynamic electricity prices, compared against a rule-based equal-share baseline.

## Main components
- Chargax environment wrapper
- Equal-share baseline
- LP controller
- Experiment runner
- Metrics and plotting pipeline

## Research goals
- Scalability with number of charge sites
- Profit vs customer satisfaction trade-offs
- Extensions: dynamic prices, battery, adaptive departures

## Status
- [X] Repository initialized
- [ ] Baseline controller
- [ ] LP controller
- [ ] Receding-horizon loop
- [ ] Benchmark suite