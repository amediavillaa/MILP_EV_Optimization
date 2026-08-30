# src/milp_ev_opt/analysis/utils.py
"""Shared utilities: styling, I/O helpers, statistical helpers, ID generation."""
from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Okabe-Ito colorblind-safe palette (8 colours)
_OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


def apply_pub_style() -> None:
    """Set publication-quality rcParams: font 12, clean spines, tight layout."""
    matplotlib.rcParams.update({
        "font.size":        12,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "figure.figsize":   (8, 5),
        "figure.dpi":       100,
        "savefig.dpi":      300,
        "figure.autolayout": True,
    })


def save_figure(fig: Figure, path: Path, dpi: int = 300) -> None:
    """Save *fig* to *path* at *dpi*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def extract_horizon(controller_name: str) -> float:
    """Return horizon int from 'milp_hN', or NaN for non-MILP controllers."""
    m = re.fullmatch(r"milp_h(\d+)", controller_name)
    return float(m.group(1)) if m else float("nan")


def format_experiment_id(
    ports: list[int],
    horizons: list[int],
    tariff: str,
    grid_cap_strain: float = 1.0,
    **_kwargs,
) -> str:
    """Build a human-readable experiment ID, e.g. 'dynamic_1_3_p3_6_h1_6'."""
    safe_tariff = re.sub(r"[^a-zA-Z0-9]", "_", tariff)
    p_str = "_".join(str(p) for p in sorted(ports))
    h_str = "_".join(str(h) for h in sorted(horizons))
    base = f"{safe_tariff}_p{p_str}_h{h_str}"
    if grid_cap_strain == 1.0:
        return base
    safe_strain = _safe_strain_str(grid_cap_strain)
    return f"{base}_s{safe_strain}"


def ci95(series: pd.Series) -> tuple[float, float]:
    """Return (ci_lower, ci_upper) 95% confidence interval for *series*."""
    n = len(series)
    mean = series.mean()
    if n < 2:
        return float(mean), float(mean)
    std = series.std(ddof=1)
    margin = 1.96 * std / np.sqrt(n)
    return float(mean - margin), float(mean + margin)


def colorblind_palette(n: int) -> list[str]:
    """Return *n* colours from the Okabe-Ito palette, cycling if n > 8."""
    return [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(n)]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_metadata(config: dict, output_dir: Path) -> None:
    """Write metadata.json to *output_dir* with reproducibility fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_id":  config.get("experiment_id", ""),
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "git_commit":     _git_hash(),
        "cli_command":    config.get("cli_command", ""),
        "hostname":       socket.gethostname(),
        "python_version": sys.version.split()[0],
        "ports":          config.get("ports", []),
        "horizons":       config.get("horizons", []),
        "n_seeds":        config.get("n_seeds", 0),
        "tariff":         config.get("tariff", ""),
        "bess_enabled":   config.get("bess_enabled", False),
        "v2g_enabled":    config.get("v2g_enabled", False),
        "solver":         config.get("solver", ""),
        "grid_cap_strain": config.get("grid_cap_strain", 1.0),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def _safe_strain_str(strain: float) -> str:
    return f"{strain:.2f}".replace(".", "_")


def validate_grid_cap_strain(values: list[float]) -> None:
    for s in values:
        if not math.isfinite(s) or not (0.0 < s <= 1.0):
            raise ValueError(
                f"--grid-cap-strain must be in (0, 1]; got {s}"
            )


def strain_save_path(save_path: str, strain: float) -> Path:
    """Return a per-strain subdirectory path.

    'results/bench.csv', 0.75  →  'results/bench_s0_75/bench.csv'
    """
    validate_grid_cap_strain([strain])
    p = Path(save_path)
    safe = _safe_strain_str(strain)
    return p.parent / f"{p.stem}_s{safe}" / p.name
