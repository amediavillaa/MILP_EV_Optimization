# Project Conventions

## Code organization
- Source code goes in `src/`
- Tests go in `tests/`
- Configurations go in `configs/`
- Documentation goes in `docs/`
- Generated outputs are not committed

## Controller interface
Each controller should expose a `compute_action(state)` method.

## Naming
- Use descriptive file names
- Keep optimization code separated into variables, constraints, objective, and solver
- Keep experiment orchestration separate from metric computation

## Branching
- `main` should remain stable
- New work should be developed in feature branches

## Outputs
Generated experiment results should be written to `outputs/` and ignored by Git unless explicitly needed