# Experiment Plan

## Initial goals
The first implementation phase focuses on establishing a working baseline and optimization pipeline.

## Phase 1
- Implement equal-share baseline
- Implement base LP controller
- Validate feasibility on small synthetic scenarios
- Run single-episode simulations

## Phase 2
- Add repeated experiments with multiple seeds
- Evaluate runtime and solution quality
- Compare LP against equal-share baseline

## Phase 3
- Scaling study by number of charge sites
- Trade-off sweeps on objective weights
- Optional extensions such as battery storage and adaptive departure handling

## Core evaluation metrics
- Total electricity cost
- Revenue / profit
- Unmet energy demand
- Fraction of satisfied charging sessions
- Average solve time
- Feasibility violations