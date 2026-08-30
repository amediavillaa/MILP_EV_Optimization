"""Shared path helpers for experiment scripts."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_results_dir(*parts: str) -> Path:
    """Return a results/ subdirectory anchored to this package's project root.

    Anchored to the package location rather than the caller's current working
    directory, so experiment output lands in a predictable place regardless
    of where the process is launched from -- e.g. once this package is nested
    inside a larger project.
    """
    return PROJECT_ROOT.joinpath("results", *parts)
