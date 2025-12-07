"""
Template driver for Shallow Water Test Case 2 (Williamson et al. 1992).

Includes:
 - Geometry / planet setup
 - Analytic IC builder
 - Norm calculations (L1, L2, Linf)
 - IOManager integration hooks (config + checkpoints)
 - Test scaffolding that can be enabled once the FV-PLR SWE solver exists
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Framework.io_manager import IOManager
from Framework.runner import run_simulation
from Solvers.geometry import CubedSphereGeometry
from Solvers.initial_conditions.swe_testcase2 import steady_geostrophic_flow
from Solvers.physics import PlanetParams
from Solvers.fv_plr_cubesphere_swe import CubedSphereSWE, SWEState


NumericalState = Dict[str, jnp.ndarray]
SolverFactory = Callable[..., "NumericalSolver"]


class NumericalSolver:
    """Minimal placeholder solver interface for the SWE driver."""

    def __init__(self, config: Dict):
        self.config = config

    def initialize(self, state: NumericalState) -> NumericalState:
        raise NotImplementedError

    def step(self, state: NumericalState, dt: float) -> NumericalState:
        raise NotImplementedError

    def state_to_output(self, state: NumericalState, output_group: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError


@dataclass
class Norms:
    l1: float
    l2: float
    linf: float


def compute_norms(field: jnp.ndarray, reference: jnp.ndarray) -> Norms:
    diff = jnp.array(field) - jnp.array(reference)
    abs_diff = jnp.abs(diff)
    l1 = float(jnp.mean(abs_diff))
    l2 = float(jnp.sqrt(jnp.mean(diff ** 2)))
    linf = float(jnp.max(abs_diff))
    return Norms(l1=l1, l2=l2, linf=linf)


def prepare_swe_config(output_dir: str = './Tests/output_swe') -> Dict:
    """Create a starter IO config for SWE runs."""
    config = {
        'solver': {'type': 'swe_testcase2'},
        'physics': {
            'R_sphere': 6.371e6,
            'gravity': 9.80616,
            'omega': 7.292115e-5,
            'name': 'Earth'
        },
        'io': {
            'output_dir': output_dir,
            'checkpoint_dir': str(Path(output_dir) / 'checkpoints'),
            'output': {
                'state': {'frequency': 10, 'enabled': True},
                'diagnostics': {'frequency': 1, 'enabled': True}
            },
            'checkpoint_frequency': 50,
            'max_checkpoints': 3
        }
    }
    return config


def run_swe_testcase2(solver_factory: SolverFactory,
                      N: int = 60,
                      days: float = 5.0,
                      save_freq_hours: float = 6.0,
                      output_dir: str = './Tests/output_swe',
                      config: Optional[Dict] = None) -> Dict[str, Norms]:
    """
    Run SWE Test Case 2 using the provided solver factory.

    Args:
        solver_factory: Callable that returns a solver instance (should inherit
            from NumericalSolver). Expected signature:
            solver_factory(N=N, geometry=geometry, planet=planet, config=config).
        N: Resolution per face.
        days: Duration of the run in days.
        save_freq_hours: Output cadence for diagnostics.
        output_dir: Directory where IOManager will write output.
        config: Optional config dictionary (will be merged with defaults).

    Returns:
        Dict mapping variable names ("h", "u_lon", "u_lat") to Norms.
    """
    if solver_factory is None:
        raise RuntimeError("SWE solver factory not provided. Supply the FV-PLR solver.")

    if config is None:
        config = prepare_swe_config(output_dir)

    geometry = CubedSphereGeometry.create(N)
    planet = PlanetParams.from_config(config)

    h_ref, u_lon_ref, u_lat_ref = steady_geostrophic_flow(geometry, planet)

    solver = solver_factory(N=N, geometry=geometry, planet=planet, config=config)
    io_manager = IOManager(solver, config)

    # Initialize solver state (assumes solver knows how to ingest analytic fields)
    state = solver.initialize({'h': jnp.array(h_ref), 'u_lon': jnp.array(u_lon_ref), 'u_lat': jnp.array(u_lat_ref)})

    dt = solver.config.get('dt', 3600.0)
    n_steps = int(days * 86400 / dt)
    steps_per_save = max(1, int((save_freq_hours * 3600) / dt))

    diagnostics = {}
    for step in range(n_steps):
        state = solver.step(state, dt)
        if step % steps_per_save == 0:
            io_manager.write_all_outputs(state)
            norms = {
                'h': compute_norms(state['h'], h_ref),
                'u_lon': compute_norms(state['u_lon'], u_lon_ref),
                'u_lat': compute_norms(state['u_lat'], u_lat_ref),
            }
            diagnostics[f'step_{state["step"]}'] = norms
    io_manager.finalize()
    return diagnostics


def plot_initial_h(h: jnp.ndarray, output_png: Path):
    h_face = np.array(h[0])
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(h_face, origin='lower', cmap='inferno')
    ax.set_title("Initial h (face 0)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="Depth [m]")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def test_swe_runner_initialization(tmp_path):
    root = Path(__file__).resolve().parents[1]
    template = root / "Config" / "swe_framework.yaml"
    config = yaml.safe_load(template.read_text())
    config["time_integration"]["num_steps"] = 0
    config["io"]["output_dir"] = str(tmp_path / "output")
    config["io"]["checkpoint_dir"] = str(tmp_path / "output" / "checkpoints")
    tmp_config = tmp_path / "swe_config.yaml"
    tmp_config.write_text(yaml.safe_dump(config))

    state = run_simulation(str(tmp_config))
    assert isinstance(state, SWEState)
    png_path = tmp_path / "initial_h.png"
    plot_initial_h(state.h, png_path)
    assert png_path.exists()


def test_run_swe_testcase2_with_solver(tmp_path):
    diagnostics = run_swe_testcase2(
        lambda **kwargs: CubedSphereSWE(
            N=kwargs["N"],
            config=kwargs["config"],
            geometry=kwargs["geometry"],
            planet=kwargs["planet"],
        ),
        N=30,
        days=1.0,
        output_dir=str(tmp_path / "run_swe"),
    )
    assert diagnostics, "Diagnostics must be recorded"

