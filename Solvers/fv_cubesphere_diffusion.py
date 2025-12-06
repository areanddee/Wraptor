"""
Cubed-Sphere Thermal Diffusion Solver

Naming convention: {numerical_method}_{grid}_{problem}.py
- fv = Finite Volume
- cubesphere = Cubed-sphere grid
- diffusion = Heat diffusion

Key features:
- Laplacian operator on cubed-sphere
- JAX sharding across 6 faces
- "Lima Flag" quadrant initial condition on Face 0
- Demonstrates cross-face connectivity

Physics: ∂T/∂t = κ ∇²T on the sphere

Refactored to use:
- geometry.CubedSphereGeometry for grid/metric
- physics.PlanetParams for physical constants
- initial_conditions for IC patterns
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import struct
import numpy as np
import yaml
from typing import Dict, Tuple, List, Any
from functools import partial
import sys
import os
import argparse

# Add paths for framework and local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))  # For halo_exchange

from Framework.solver_interface import NumericalSolver

# Import new modular components
from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams, EARTH
from Solvers.initial_conditions import lima_flag

# Import optimized halo exchange (same directory)
from halo_exchange import (
    create_communication_schedule,
    make_halo_exchange,
    exchange_scalar_halos_v2,
    extend_to_include_ghosts
)


# ============================================================================
# STATE
# ============================================================================

@struct.dataclass
class DiffusionState:
    """State for cubed-sphere thermal diffusion"""
    T: jax.Array          # (6, N, N) - temperature field on all faces (interior only)
    time: float
    step: int


# ============================================================================
# GEOMETRY (LEGACY - kept for backward compatibility, use geometry module)
# ============================================================================

def equiangular_to_xyz_face(xi1, xi2, face_id, R=6.371e6):
    """Map equiangular (ξ¹, ξ²) to Cartesian (X, Y, Z).
    
    LEGACY: Use Solvers.geometry.CubedSphereGeometry for new code.
    """
    tan_xi1 = jnp.tan(xi1)
    tan_xi2 = jnp.tan(xi2)
    delta = jnp.sqrt(1.0 + tan_xi1**2 + tan_xi2**2)
    
    if face_id == 0:  # +Z (north pole)
        X, Y, Z = R * tan_xi1 / delta, R * tan_xi2 / delta, R / delta
    elif face_id == 1:  # +Y
        X, Y, Z = -R * tan_xi1 / delta, R / delta, R * tan_xi2 / delta
    elif face_id == 2:  # -X
        X, Y, Z = -R / delta, -R * tan_xi1 / delta, R * tan_xi2 / delta
    elif face_id == 3:  # -Y
        X, Y, Z = R * tan_xi1 / delta, -R / delta, R * tan_xi2 / delta
    elif face_id == 4:  # +X
        X, Y, Z = R / delta, R * tan_xi1 / delta, R * tan_xi2 / delta
    elif face_id == 5:  # -Z (south pole)
        X, Y, Z = -R * tan_xi1 / delta, R * tan_xi2 / delta, -R / delta
    
    return X, Y, Z


def compute_metric_face(xi1, xi2, R=1.0):
    """Compute √G for given face.
    
    LEGACY: Use Solvers.geometry.CubedSphereGeometry for new code.
    Note: Now defaults to R=1.0 (unit sphere). Scale result by R² if needed.
    """
    tan_x1 = jnp.tan(xi1)
    tan_x2 = jnp.tan(xi2)
    cos_x1 = jnp.cos(xi1)
    cos_x2 = jnp.cos(xi2)
    delta = jnp.sqrt(1.0 + tan_x1**2 + tan_x2**2)
    sqrtG = R**2 / (delta**3 * cos_x1**2 * cos_x2**2)
    return sqrtG


# ============================================================================
# DIFFUSION OPERATOR
# ============================================================================

def compute_diffusion_rhs(T_ghosts, sqrtG_interior, kappa, dx, N, R_sphere=6.371e6):
    """
    Compute RHS of diffusion equation using ghost cells.
    
    ∂T/∂t = κ ∇²T
    
    where ∇²T = (1/√G) [∂/∂ξ¹(√G ∂T/∂ξ¹) + ∂/∂ξ²(√G ∂T/∂ξ²)]
    
    Simplified: ∇²T ≈ ∂²T/∂x² + ∂²T/∂y² (neglecting metric variations)
    
    Args:
        T_ghosts: (6, N+2, N+2) with filled ghosts
        sqrtG_interior: (6, N, N) metric on interior only
        kappa: Diffusion coefficient [m²/s]
        dx: Grid spacing [radians]
        N: Interior resolution
        R_sphere: Planet radius [m] (default: Earth)
        
    Returns:
        rhs: (6, N, N) tendency ∂T/∂t on interior [K/s]
    """
    rhs_all = jnp.zeros((6, N, N))
    
    # Convert dx from radians to meters for physical diffusion
    dx_physical = dx * R_sphere
    
    for face in range(6):
        T = T_ghosts[face]  # (N+2, N+2)
        
        # Second derivatives using 3-point stencil in physical coordinates
        # ∂²T/∂x²
        d2T_dx1 = (T[2:N+2, 1:N+1] - 2*T[1:N+1, 1:N+1] + T[0:N, 1:N+1]) / dx_physical**2
        
        # ∂²T/∂y²
        d2T_dx2 = (T[1:N+1, 2:N+2] - 2*T[1:N+1, 1:N+1] + T[1:N+1, 0:N]) / dx_physical**2
        
        # Simple Laplacian (neglecting metric variations for simplicity)
        laplacian = d2T_dx1 + d2T_dx2
        
        # RHS: κ ∇²T
        rhs = kappa * laplacian
        rhs_all = rhs_all.at[face].set(rhs)
    
    return rhs_all


# ============================================================================
# TIME INTEGRATION
# ============================================================================

def euler_step_compiled(state: DiffusionState, dx: float, dt: float, kappa: float,
                       N: int, sqrtG_all: jax.Array, 
                       halo_exchange_fn, R_sphere: float = 6.371e6) -> DiffusionState:
    """
    Single forward Euler timestep - JIT compiled.
    
    Args:
        state: Current diffusion state
        dx: Grid spacing [rad]
        dt: Timestep [s]
        kappa: Diffusion coefficient [m²/s]
        N: Grid resolution
        sqrtG_all: (6, N, N) metric
        halo_exchange_fn: Pre-compiled halo exchange function
        R_sphere: Planet radius [m]
        
    Returns:
        new_state: Updated state
    """
    # Exchange halos using optimized v2 wrapper
    T_ghosts = exchange_scalar_halos_v2(state.T, N, halo_exchange_fn)
    
    # Compute RHS
    rhs = compute_diffusion_rhs(T_ghosts, sqrtG_all, kappa, dx, N, R_sphere)
    
    # Update temperature
    T_new = state.T + dt * rhs
    
    return DiffusionState(T=T_new, time=state.time + dt, step=state.step + 1)


# ============================================================================
# SOLVER CLASS
# ============================================================================

class CubedSphereDiffusion(NumericalSolver):
    """
    Thermal diffusion solver on cubed-sphere grid.
    
    Solves: ∂T/∂t = κ ∇²T
    
    Features:
    - Forward Euler time integration
    - Lima Flag quadrant initial condition
    - Optimized halo exchange (functools.partial + JIT)
    - Optional JAX sharding for multi-device
    """
    
    def __init__(self, N: int, kappa: float = 5e5, config_file: str = None):
        """
        Initialize solver.
        
        Args:
            N: Grid resolution (N×N per face)
            kappa: Diffusion coefficient [m²/s]
            config_file: Path to YAML config (for parallelization settings)
        """
        self.N = N
        self.kappa = float(kappa)  # Ensure kappa is float (may come from YAML as string)
        
        # Load config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.config = config
            print(f"✓ Loaded config: {config_file}")
        else:
            self.config = {'parallelization': {'enable_sharding': False}}
            if config_file:
                print(f"⚠ Config not found: {config_file}, using defaults")
        
        # Create geometry (unit sphere, always f64)
        self.geometry = CubedSphereGeometry.create(N)
        self.dx = self.geometry.dx
        
        # Get planet parameters from config (SINGLE SOURCE OF TRUTH)
        self.planet = PlanetParams.from_config(self.config)
        
        # Setup grid
        print(f"\nSolver configuration:")
        print(f"  Grid: {N}×{N} per face (6 faces, {6*N*N} total cells)")
        print(f"  Method: Forward Euler diffusion")
        print(f"  dx: {self.dx:.6f} rad ({self.dx * self.planet.R_sphere / 1e3:.1f} km)")
        print(f"  κ (diffusivity): {self.kappa:.2e} m²/s")
        print(f"  Planet: {self.planet.name} (R={self.planet.R_sphere/1e6:.3f}×10⁶ m)")
        
        # Compute timestep constraint
        dx_physical = self.dx * self.planet.R_sphere
        dt_cfl_max = 0.25 * dx_physical**2 / self.kappa
        print(f"  dt_max (CFL): {dt_cfl_max:.1f} s ({dt_cfl_max/60:.1f} min)")
        
        # Create halo exchange with functools.partial + JIT
        print(f"\nCompiling halo exchange (functools.partial + JIT)...")
        schedule = create_communication_schedule()
        self.halo_exchange_fn = make_halo_exchange(schedule, N)
        print(f"  ✓ Halo exchange compiled (12 edge swaps, 0 recompilation)")
        
        # Setup sharding if requested
        if self.config.get('parallelization', {}).get('enable_sharding', False):
            self.setup_sharding()
        else:
            self.sharding = None
            print(f"  Sharding: Disabled (single device)")
        
        # Pre-compute geometry
        self.setup_geometry()
        
        print(f"\n✓ Solver ready!")
    
    def setup_sharding(self):
        """Setup JAX sharding for multi-device execution."""
        para_config = self.config['parallelization']
        device_type = para_config.get('device_type', 'cpu')
        num_devices = para_config.get('num_devices', 6)
        tiles_per_edge = para_config.get('tiles_per_edge', 1)
        
        print(f"\n{'='*70}")
        print(f"SETTING UP JAX SHARDING")
        print(f"{'='*70}")
        
        # Validate tiles_per_edge
        if tiles_per_edge != 1:
            raise NotImplementedError(
                f"Error: tiles_per_edge = {tiles_per_edge} is not yet supported.\n"
                f"Currently only tiles_per_edge = 1 is implemented (6 tiles total).\n"
                f"Future work will enable multi-tile-per-face for scaling to 100+ GPUs.\n"
                f"Example: tiles_per_edge = 3 → 54 tiles → run on 56 B200 chips."
            )
        
        # Calculate total tiles
        num_tiles = 6 * tiles_per_edge * tiles_per_edge
        
        # Validate device count
        if num_devices > num_tiles:
            raise ValueError(
                f"Error: num_devices = {num_devices} exceeds num_tiles = {num_tiles}.\n"
                f"Cannot have more devices than tiles.\n"
                f"With tiles_per_edge = {tiles_per_edge}, max devices = {num_tiles}."
            )
        
        if num_tiles % num_devices != 0:
            valid_counts = [d for d in range(1, num_tiles + 1) if num_tiles % d == 0]
            raise ValueError(
                f"Error: num_tiles = {num_tiles} is not evenly divisible by num_devices = {num_devices}.\n"
                f"JAX requires: num_tiles % num_devices == 0\n"
                f"With tiles_per_edge = {tiles_per_edge}, valid device counts are:\n"
                f"  {valid_counts}"
            )
        
        print(f"  Tile configuration:")
        print(f"    tiles_per_edge: {tiles_per_edge}")
        print(f"    total tiles: {num_tiles} (6 faces × {tiles_per_edge}² tiles/face)")
        print(f"    tiles per device: {num_tiles / num_devices:.1f}")
        
        if device_type == 'cpu':
            # Create virtual devices for CPU testing
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'
            print(f"  Device type: CPU (virtual devices)")
            print(f"  XLA_FLAGS set: {num_devices} virtual CPU devices")
        else:
            print(f"  Device type: GPU")
            print(f"  Requested devices: {num_devices}")
        
        devices = jax.devices()[:num_devices]
        print(f"  Available devices: {len(jax.devices())}")
        print(f"  Using devices: {devices}")
        
        # Create mesh: 1D array of devices mapped to 'tiles' axis
        self.mesh = Mesh(devices, ('tiles',))
        self.sharding = NamedSharding(self.mesh, P('tiles'))
        
        print(f"  Mesh created: {num_tiles} tiles across {len(devices)} device(s)")
        print(f"  Sharding strategy: PartitionSpec('tiles') on axis 0")
        if num_tiles > len(devices):
            print(f"  Note: Multiple tiles per device (serial execution)")
        print(f"{'='*70}\n")
    
    def setup_geometry(self):
        """Pre-compute metric arrays from geometry module."""
        # Use geometry module's sqrtG (computed on unit sphere, f64)
        # Scale by R² for physical metric
        R = self.planet.R_sphere
        self.sqrtG_all = jnp.array(self.geometry.sqrtG) * R**2
    
    def initialize(self, pattern='quadrant', T_hot=600.0, T_background=1.0) -> DiffusionState:
        """
        Initialize state.
        
        Args:
            pattern: 'quadrant' (Lima Flag on Face 0)
            T_hot: Hot temperature [K]
            T_background: Background temperature [K]
        """
        # Use IC module for Lima Flag pattern
        if pattern == 'quadrant':
            T_all = lima_flag(self.geometry, T_hot=T_hot, T_background=T_background)
            print(f"\nInitial condition: Lima Flag quadrant pattern")
            print(f"  Face 0 (north pole): Upper-left & lower-right at {T_hot}K")
            print(f"  All other faces: {T_background}K")
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Apply sharding if enabled
        if self.sharding is not None:
            T_all = jax.device_put(T_all, self.sharding)
        
        return DiffusionState(T=T_all, time=0.0, step=0)
    
    def step(self, state: DiffusionState, dt: float) -> DiffusionState:
        """Advance one timestep."""
        return euler_step_compiled(state, self.dx, dt, self.kappa, self.N,
                                  self.sqrtG_all, self.halo_exchange_fn,
                                  self.planet.R_sphere)
    
    def get_diagnostics(self, state: DiffusionState) -> Dict[str, float]:
        """Compute diagnostics."""
        T_max = float(jnp.max(state.T))
        T_sum = float(jnp.sum(state.T * self.sqrtG_all * self.dx**2))
        
        # Per-face diagnostics
        T_face0_max = float(jnp.max(state.T[0]))
        T_equatorial_max = float(jnp.max(state.T[1:5]))
        
        return {
            'T_max': T_max,
            'heat_content': T_sum,
            'T_face0_max': T_face0_max,
            'T_equatorial_max': T_equatorial_max,
            'time': state.time,
            'step': state.step
        }
    
    # ========================================================================
    # FRAMEWORK INTERFACE METHODS
    # ========================================================================
    
    def get_available_outputs(self) -> Dict[str, List[str]]:
        """Define available output groups and their variables."""
        return {
            'state': ['T'],  # Temperature field for visualization/restart
            'diagnostics': ['T_max', 'heat_content', 'T_face0_max', 'T_equatorial_max']
        }
    
    def get_output_spec(self, config: Dict[str, Any]) -> Dict[str, 'OutputSpec']:
        """Create output specifications from config."""
        from Framework.solver_interface import OutputSpec
        
        io_config = config['io']['output']
        available = self.get_available_outputs()
        
        return {
            'state': OutputSpec(
                variables=available['state'],
                frequency=io_config['state']['frequency'],
                enabled=io_config['state']['enabled'],
                description="Temperature field on all 6 faces"
            ),
            'diagnostics': OutputSpec(
                variables=available['diagnostics'],
                frequency=io_config['diagnostics']['frequency'],
                enabled=io_config['diagnostics']['enabled'],
                description="Diagnostic scalars (max T, heat content, etc.)"
            )
        }
    
    def state_to_output(self, state: DiffusionState, 
                       output_group: str = 'state') -> Dict[str, np.ndarray]:
        """
        Convert state to output format for a specific group.
        
        Args:
            state: Current diffusion state
            output_group: 'state' or 'diagnostics'
        """
        if output_group == 'state':
            return {
                'T': np.array(state.T),
                'time': state.time,
                'step': state.step
            }
        elif output_group == 'diagnostics':
            return self.get_diagnostics(state)
        else:
            raise ValueError(f"Unknown output group: {output_group}")
    
    def state_from_checkpoint(self, checkpoint_data: Dict[str, np.ndarray]) -> DiffusionState:
        """
        Restore state from checkpoint data.
        
        Args:
            checkpoint_data: Dictionary containing saved state arrays
        
        Returns:
            Restored diffusion state
        """
        T = jnp.array(checkpoint_data['T'])
        time = float(checkpoint_data['time'])
        step = int(checkpoint_data['step'])
        
        # Apply sharding if solver was initialized with it
        if hasattr(self, 'sharding') and self.sharding is not None:
            T = jax.device_put(T, self.sharding)
        
        return DiffusionState(T=T, time=time, step=step)


# ============================================================================
# JIT-COMPILED STEP FUNCTION
# ============================================================================

# Global JIT-compiled step function (created on first use)
_euler_step_jit = jax.jit(euler_step_compiled, 
                          static_argnames=['N'])


# ============================================================================
# STANDALONE TEST
# ============================================================================

def run_standalone_test(N=30, days=30.0, kappa=5e5, config_file=None):
    """
    Standalone test function.
    
    Args:
        N: Grid resolution
        days: Simulation length [days]
        kappa: Diffusion coefficient [m²/s]
        config_file: Path to config YAML
    """
    print("="*70)
    print("STANDALONE THERMAL DIFFUSION TEST")
    print("="*70)
    print(f"Resolution: {N}×{N} per face")
    print(f"Duration: {days} days")
    print(f"Diffusivity: κ = {kappa:.2e} m²/s")
    
    # Create solver
    solver = CubedSphereDiffusion(N=N, kappa=kappa, config_file=config_file)
    
    # Initialize
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    # Time integration setup
    dt = 1800.0  # 30 minutes
    total_time = days * 86400.0
    n_steps = int(total_time / dt)
    
    # Check CFL stability
    dx_physical = solver.dx * solver.planet.R_sphere
    dt_cfl_max = 0.25 * dx_physical**2 / kappa
    cfl_number = dt / dt_cfl_max
    
    print(f"\nTime integration:")
    print(f"  dt: {dt:.1f} s ({dt/60:.1f} min)")
    print(f"  CFL: {cfl_number:.3f} {'(STABLE)' if cfl_number <= 0.5 else '(WARNING: May be unstable)'}")
    print(f"  n_steps: {n_steps}")
    
    # Compile step function
    print(f"\nJIT compiling step function...")
    state = solver.step(state, dt)
    print(f"  ✓ Compilation complete")
    
    # Save frequency: every 5 days
    save_freq = int((5 * 86400) / dt)
    
    # Time integration
    print(f"\nRunning {n_steps} steps (saving every {save_freq})...")
    
    initial_heat = solver.get_diagnostics(state)['heat_content']
    
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        if (step + 1) % save_freq == 0 or step == 0:
            diag = solver.get_diagnostics(state)
            heat_error = (diag['heat_content'] - initial_heat) / initial_heat
            
            print(f"  Step {step+1:4d} (day {diag['time']/86400:5.2f}): "
                  f"T_max={diag['T_max']:6.1f}K, "
                  f"Face0_max={diag['T_face0_max']:6.1f}K, "
                  f"Eq_max={diag['T_equatorial_max']:6.1f}K, "
                  f"Heat_err={heat_error:.2e}")
    
    # Final diagnostics
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    final_diag = solver.get_diagnostics(state)
    heat_error_final = (final_diag['heat_content'] - initial_heat) / initial_heat
    
    print(f"Heat conservation:")
    print(f"  Initial: {initial_heat:.6e}")
    print(f"  Final:   {final_diag['heat_content']:.6e}")
    print(f"  Error:   {heat_error_final:.2e}")
    print(f"\nTemperature evolution:")
    print(f"  Initial T_max: 600.0 K")
    print(f"  Final T_max:   {final_diag['T_max']:.1f} K")
    print(f"{'='*70}\n")
    
    print("✓ Test complete!")
    
    return state, solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cubed-Sphere Thermal Diffusion Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-size', type=int, default=30,
                        help='Grid resolution (N×N cells per face)')
    parser.add_argument('--days', type=float, default=30.0,
                        help='Simulation length in days')
    parser.add_argument('--kappa', type=float, default=5e5,
                        help='Diffusion coefficient [m²/s]')
    parser.add_argument('--config', type=str,
                        default='../Config/config_diffusion.yaml',
                        help='Path to config file (for parallelization)')
    
    args = parser.parse_args()
    
    run_standalone_test(N=args.grid_size, days=args.days, 
                       kappa=args.kappa, config_file=args.config)

