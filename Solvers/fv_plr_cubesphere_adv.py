"""
Cubed-Sphere Advection Solver with Piecewise Linear Reconstruction (PLR)

Naming convention: {numerical_method}_{grid}_{problem}.py
- fv = Finite Volume
- plr = Piecewise Linear Reconstruction (2nd-order)
- cubesphere = Cubed-sphere grid
- adv = Advection

Key features:
- 2nd-order accurate PLR with MC slope limiter
- Cartesian velocity exchange (no rotation needed!)
- JAX sharding across 6 faces
- Williamson et al. (1992) Test Case 1: Solid body rotation

Physics: ∂q/∂t + ∇·(q V) = 0 on the sphere

Refactored to use:
- geometry.CubedSphereGeometry for grid/metric
- physics.PlanetParams for physical constants
- initial_conditions for IC patterns and velocity fields
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
from Solvers.initial_conditions import cosine_bell, solid_body_rotation

# Import halo exchange (same directory)
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
class AdvectionState:
    """State for cubed-sphere advection"""
    q: jax.Array          # (6, N, N) - scalar field on all faces (interior only)
    Vx: jax.Array         # (6, N, N) - Cartesian velocity x-component
    Vy: jax.Array         # (6, N, N) - Cartesian velocity y-component  
    Vz: jax.Array         # (6, N, N) - Cartesian velocity z-component
    time: float
    step: int


# ============================================================================
# COORDINATE TRANSFORMATIONS (LEGACY - use geometry module for new code)
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


def xyz_to_lonlat(X, Y, Z, R=1.0):
    """Convert Cartesian to (lon, lat).
    
    Note: For unit sphere, R=1.0. For physical coords, pass R_sphere.
    """
    lon = jnp.arctan2(Y, X)
    # Use magnitude of position vector for latitude calculation
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    lat = jnp.arcsin(Z / r)
    return lon, lat


def compute_jacobian_face(xi1, xi2, face_id, R=6.371e6):
    """Compute Jacobian for given face."""
    tan_x1, tan_x2 = jnp.tan(xi1), jnp.tan(xi2)
    sec2_x1, sec2_x2 = 1.0 / jnp.cos(xi1)**2, 1.0 / jnp.cos(xi2)**2
    
    delta = jnp.sqrt(1.0 + tan_x1**2 + tan_x2**2)
    delta3 = delta**3
    
    term1 = R * sec2_x1 * (1.0 + tan_x2**2) / delta3
    term2 = -R * tan_x1 * tan_x2 * sec2_x2 / delta3
    term3 = -R * tan_x1 * tan_x2 * sec2_x1 / delta3
    term4 = R * sec2_x2 * (1.0 + tan_x1**2) / delta3
    term5 = -R * tan_x1 * sec2_x1 / delta3
    term6 = -R * tan_x2 * sec2_x2 / delta3
    
    if face_id == 0:
        J11, J12, J21, J22, J31, J32 = term1, term2, term3, term4, term5, term6
    elif face_id == 1:
        J11, J12, J21, J22, J31, J32 = -term1, -term2, term5, term6, term3, term4
    elif face_id == 2:
        J11, J12, J21, J22, J31, J32 = -term5, -term6, -term1, -term2, term3, term4
    elif face_id == 3:
        J11, J12, J21, J22, J31, J32 = term1, term2, -term5, -term6, term3, term4
    elif face_id == 4:
        J11, J12, J21, J22, J31, J32 = term5, term6, term1, term2, term3, term4
    elif face_id == 5:
        J11, J12, J21, J22, J31, J32 = -term1, -term2, term3, term4, -term5, -term6
    
    return J11, J12, J21, J22, J31, J32


def contravariant_to_cartesian(u1, u2, xi1, xi2, face_id):
    """Transform contravariant (u¹, u²) to Cartesian (Vx, Vy, Vz)."""
    J11, J12, J21, J22, J31, J32 = compute_jacobian_face(xi1, xi2, face_id)
    V_x = J11 * u1 + J12 * u2
    V_y = J21 * u1 + J22 * u2
    V_z = J31 * u1 + J32 * u2
    return V_x, V_y, V_z


def cartesian_to_contravariant(V_x, V_y, V_z, xi1, xi2, face_id):
    """Transform Cartesian to contravariant using pseudoinverse."""
    J11, J12, J21, J22, J31, J32 = compute_jacobian_face(xi1, xi2, face_id)
    
    # (J^T J)^{-1}
    JTJ_11 = J11**2 + J21**2 + J31**2
    JTJ_12 = J11*J12 + J21*J22 + J31*J32
    JTJ_22 = J12**2 + J22**2 + J32**2
    det_JTJ = JTJ_11 * JTJ_22 - JTJ_12**2
    
    JTJ_inv_11 = JTJ_22 / det_JTJ
    JTJ_inv_12 = -JTJ_12 / det_JTJ
    JTJ_inv_22 = JTJ_11 / det_JTJ
    
    # J^T · V
    JT_V_1 = J11*V_x + J21*V_y + J31*V_z
    JT_V_2 = J12*V_x + J22*V_y + J32*V_z
    
    # u = (J^T J)^{-1} · (J^T · V)
    u1 = JTJ_inv_11 * JT_V_1 + JTJ_inv_12 * JT_V_2
    u2 = JTJ_inv_12 * JT_V_1 + JTJ_inv_22 * JT_V_2
    
    return u1, u2


def compute_metric_face(xi1, xi2, R=1.0):
    """Compute √G.
    
    LEGACY: Use Solvers.geometry.CubedSphereGeometry for new code.
    Note: Now defaults to R=1.0 (unit sphere). Scale result by R² if needed.
    """
    tan_x1, tan_x2 = jnp.tan(xi1), jnp.tan(xi2)
    cos_x1, cos_x2 = jnp.cos(xi1), jnp.cos(xi2)
    delta = jnp.sqrt(1.0 + tan_x1**2 + tan_x2**2)
    return R**2 / (delta**3 * cos_x1**2 * cos_x2**2)


# ============================================================================
# INITIAL CONDITIONS (LEGACY - use initial_conditions module for new code)
# ============================================================================

def great_circle_distance(lon1, lat1, lon2, lat2, R=6.371e6):
    """Great circle distance.
    
    LEGACY: Use Solvers.initial_conditions.cosine_bell for new code.
    """
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = jnp.sin(dlat/2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon/2)**2
    return R * 2 * jnp.arcsin(jnp.sqrt(a))


def cosine_bell_legacy(lon, lat, lon_c, lat_c, h0=1000.0, R_sphere=6.371e6):
    """Cosine bell (Williamson et al. 1994).
    
    LEGACY: Use Solvers.initial_conditions.cosine_bell for new code.
    """
    r = great_circle_distance(lon, lat, lon_c, lat_c, R=R_sphere)
    r_bell = R_sphere / 3.0
    return jnp.where(r < r_bell,
                     h0 * 0.5 * (1.0 + jnp.cos(jnp.pi * r / r_bell)),
                     0.0)


def initialize_cosine_bell(N, lon_c=3*jnp.pi/2, lat_c=0.0):
    """Initialize cosine bell on all 6 faces."""
    xi1_1d = jnp.linspace(-jnp.pi/4, jnp.pi/4, N)
    xi2_1d = jnp.linspace(-jnp.pi/4, jnp.pi/4, N)
    XI1, XI2 = jnp.meshgrid(xi1_1d, xi2_1d, indexing='ij')
    
    q_all = jnp.zeros((6, N, N))
    
    for face in range(6):
        X, Y, Z = equiangular_to_xyz_face(XI1, XI2, face)
        lon, lat = xyz_to_lonlat(X, Y, Z)
        q_all = q_all.at[face].set(cosine_bell(lon, lat, lon_c, lat_c))
    
    return q_all


# ============================================================================
# VELOCITY FIELDS
# ============================================================================

def solid_body_rotation_velocity(XI1, XI2, face_id, u0):
    """Williamson Test Case 1: Solid body rotation."""
    X, Y, Z = equiangular_to_xyz_face(XI1, XI2, face_id)
    lon, lat = xyz_to_lonlat(X, Y, Z)
    
    # Spherical velocity
    u_lon = u0 * jnp.cos(lat)
    u_lat = 0.0
    
    # Convert to Cartesian
    V_x = -u_lon * jnp.sin(lon) - u_lat * jnp.sin(lat) * jnp.cos(lon)
    V_y = u_lon * jnp.cos(lon) - u_lat * jnp.sin(lat) * jnp.sin(lon)
    V_z = u_lat * jnp.cos(lat)
    
    return V_x, V_y, V_z


# ============================================================================
# PLR SLOPE LIMITERS (import from working code)
# ============================================================================

def minmod(a, b):
    """Minmod limiter."""
    return jnp.where(a * b > 0,
                     jnp.where(jnp.abs(a) < jnp.abs(b), a, b),
                     0.0)

def minmod3(a, b, c):
    """Three-argument minmod."""
    return minmod(a, minmod(b, c))

def mc_limiter(central, forward, backward):
    """Monotonized Central (MC) limiter."""
    return minmod3(2.0 * forward, 2.0 * backward, central)

def compute_limited_slopes(q_ghosts, dx, limiter='MC'):
    """Compute limited slopes for PLR."""
    q_interior = q_ghosts[1:-1, 1:-1]
    
    # ξ¹ direction
    q_im1, q_ip1 = q_ghosts[0:-2, 1:-1], q_ghosts[2:, 1:-1]
    slope1_central = (q_ip1 - q_im1) / (2.0 * dx)
    slope1_forward = (q_ip1 - q_interior) / dx
    slope1_backward = (q_interior - q_im1) / dx
    
    # ξ² direction
    q_jm1, q_jp1 = q_ghosts[1:-1, 0:-2], q_ghosts[1:-1, 2:]
    slope2_central = (q_jp1 - q_jm1) / (2.0 * dx)
    slope2_forward = (q_jp1 - q_interior) / dx
    slope2_backward = (q_interior - q_jm1) / dx
    
    if limiter == 'MC':
        slope1 = mc_limiter(slope1_central, slope1_forward, slope1_backward)
        slope2 = mc_limiter(slope2_central, slope2_forward, slope2_backward)
    else:
        slope1 = minmod3(slope1_forward, slope1_backward, slope1_central)
        slope2 = minmod3(slope2_forward, slope2_backward, slope2_central)
    
    return slope1, slope2


# ============================================================================
# FLUX COMPUTATION (PLR - 2nd order)
# ============================================================================

def compute_flux_plr(q_ghosts, Vx_ghosts, Vy_ghosts, Vz_ghosts,
                     XI1_ghosts, XI2_ghosts, sqrtG_ghosts, face_id, dx, N):
    """Compute PLR flux using Cartesian velocities."""
    # Compute limited slopes
    slope1, slope2 = compute_limited_slopes(q_ghosts, dx, limiter='MC')
    q_interior = q_ghosts[1:-1, 1:-1]
    
    # ξ¹ direction flux
    q_right_edge_interior = q_interior + 0.5 * dx * slope1
    q_left_edge_interior = q_interior - 0.5 * dx * slope1
    
    q_right_edge = jnp.concatenate([
        q_ghosts[0:1, 1:N+1],
        q_right_edge_interior,
        q_ghosts[N+1:N+2, 1:N+1]
    ], axis=0)
    
    q_left_edge = jnp.concatenate([
        q_ghosts[0:1, 1:N+1],
        q_left_edge_interior,
        q_ghosts[N+1:N+2, 1:N+1]
    ], axis=0)
    
    # Average Cartesian velocities to faces
    Vx_face = 0.5 * (Vx_ghosts[1:N+2, 1:N+1] + Vx_ghosts[0:N+1, 1:N+1])
    Vy_face = 0.5 * (Vy_ghosts[1:N+2, 1:N+1] + Vy_ghosts[0:N+1, 1:N+1])
    Vz_face = 0.5 * (Vz_ghosts[1:N+2, 1:N+1] + Vz_ghosts[0:N+1, 1:N+1])
    
    xi1_face = 0.5 * (XI1_ghosts[1:N+2, 1:N+1] + XI1_ghosts[0:N+1, 1:N+1])
    xi2_face = 0.5 * (XI2_ghosts[1:N+2, 1:N+1] + XI2_ghosts[0:N+1, 1:N+1])
    
    u1_face, _ = cartesian_to_contravariant(Vx_face, Vy_face, Vz_face,
                                            xi1_face, xi2_face, face_id)
    
    sqrtG_face = 0.5 * (sqrtG_ghosts[1:N+2, 1:N+1] + sqrtG_ghosts[0:N+1, 1:N+1])
    
    q_face = jnp.where(u1_face > 0,
                       q_right_edge[0:N+1, :],
                       q_left_edge[1:N+2, :])
    
    F1 = sqrtG_face * u1_face * q_face
    
    # ξ² direction flux
    q_top_edge_interior = q_interior + 0.5 * dx * slope2
    q_bottom_edge_interior = q_interior - 0.5 * dx * slope2
    
    q_top_edge = jnp.concatenate([
        q_ghosts[1:N+1, 0:1],
        q_top_edge_interior,
        q_ghosts[1:N+1, N+1:N+2]
    ], axis=1)
    
    q_bottom_edge = jnp.concatenate([
        q_ghosts[1:N+1, 0:1],
        q_bottom_edge_interior,
        q_ghosts[1:N+1, N+1:N+2]
    ], axis=1)
    
    Vx_face2 = 0.5 * (Vx_ghosts[1:N+1, 1:N+2] + Vx_ghosts[1:N+1, 0:N+1])
    Vy_face2 = 0.5 * (Vy_ghosts[1:N+1, 1:N+2] + Vy_ghosts[1:N+1, 0:N+1])
    Vz_face2 = 0.5 * (Vz_ghosts[1:N+1, 1:N+2] + Vz_ghosts[1:N+1, 0:N+1])
    
    xi1_face2 = 0.5 * (XI1_ghosts[1:N+1, 1:N+2] + XI1_ghosts[1:N+1, 0:N+1])
    xi2_face2 = 0.5 * (XI2_ghosts[1:N+1, 1:N+2] + XI2_ghosts[1:N+1, 0:N+1])
    
    _, u2_face = cartesian_to_contravariant(Vx_face2, Vy_face2, Vz_face2,
                                            xi1_face2, xi2_face2, face_id)
    
    sqrtG_face2 = 0.5 * (sqrtG_ghosts[1:N+1, 1:N+2] + sqrtG_ghosts[1:N+1, 0:N+1])
    
    q_face2 = jnp.where(u2_face > 0,
                        q_top_edge[:, 0:N+1],
                        q_bottom_edge[:, 1:N+2])
    
    F2 = sqrtG_face2 * u2_face * q_face2
    
    return F1, F2


def compute_rhs_plr(q_ghosts, Vx_ghosts, Vy_ghosts, Vz_ghosts,
                    XI1_ghosts, XI2_ghosts, sqrtG_ghosts, face_id, dx, N):
    """Compute RHS with PLR."""
    F1, F2 = compute_flux_plr(q_ghosts, Vx_ghosts, Vy_ghosts, Vz_ghosts,
                              XI1_ghosts, XI2_ghosts, sqrtG_ghosts,
                              face_id, dx, N)
    
    dF1_dx1 = (F1[1:, :] - F1[:-1, :]) / dx
    dF2_dx2 = (F2[:, 1:] - F2[:, :-1]) / dx
    div_F = dF1_dx1 + dF2_dx2
    
    sqrtG_interior = sqrtG_ghosts[1:N+1, 1:N+1]
    return -div_F / sqrtG_interior


# ============================================================================
# TIME INTEGRATION
# ============================================================================

def rk3_step(state: AdvectionState, dx: float, dt: float, N: int,
             XI1_all, XI2_all, sqrtG_all, halo_exchange_fn) -> AdvectionState:
    """RK3 step with PLR on all faces."""
    q0 = state.q
    
    # Extend geometry (doesn't need proper halo exchange)
    XI1_ghosts = jnp.pad(XI1_all, ((0,0), (1,1), (1,1)), mode='edge')
    XI2_ghosts = jnp.pad(XI2_all, ((0,0), (1,1), (1,1)), mode='edge')
    sqrtG_ghosts = jnp.pad(sqrtG_all, ((0,0), (1,1), (1,1)), mode='edge')
    
    # Helper to extend + exchange
    def exchange_with_ghosts(field_interior):
        """Extend (6,N,N) → (6,N+2,N+2) and exchange halos."""
        return exchange_scalar_halos_v2(field_interior, N, halo_exchange_fn)
    
    # Stage 1
    q_ghosts = exchange_with_ghosts(state.q)
    Vx_ghosts = exchange_with_ghosts(state.Vx)
    Vy_ghosts = exchange_with_ghosts(state.Vy)
    Vz_ghosts = exchange_with_ghosts(state.Vz)
    
    q1 = jnp.zeros_like(q0)
    for face in range(6):
        rhs = compute_rhs_plr(q_ghosts[face], Vx_ghosts[face], Vy_ghosts[face], Vz_ghosts[face],
                             XI1_ghosts[face], XI2_ghosts[face], sqrtG_ghosts[face],
                             face, dx, N)
        q1 = q1.at[face].set(q0[face] + dt * rhs)
    
    # Stage 2
    q_ghosts = exchange_with_ghosts(q1)
    
    q2 = jnp.zeros_like(q0)
    for face in range(6):
        rhs = compute_rhs_plr(q_ghosts[face], Vx_ghosts[face], Vy_ghosts[face], Vz_ghosts[face],
                             XI1_ghosts[face], XI2_ghosts[face], sqrtG_ghosts[face],
                             face, dx, N)
        q2 = q2.at[face].set(0.75*q0[face] + 0.25*(q1[face] + dt*rhs))
    
    # Stage 3
    q_ghosts = exchange_with_ghosts(q2)
    
    q3 = jnp.zeros_like(q0)
    for face in range(6):
        rhs = compute_rhs_plr(q_ghosts[face], Vx_ghosts[face], Vy_ghosts[face], Vz_ghosts[face],
                             XI1_ghosts[face], XI2_ghosts[face], sqrtG_ghosts[face],
                             face, dx, N)
        q3 = q3.at[face].set((1.0/3.0)*q0[face] + (2.0/3.0)*(q2[face] + dt*rhs))
    
    return AdvectionState(
        q=q3,
        Vx=state.Vx,  # Velocities don't change
        Vy=state.Vy,
        Vz=state.Vz,
        time=state.time + dt,
        step=state.step + 1
    )


rk3_step_compiled = jax.jit(rk3_step, static_argnames=['N', 'halo_exchange_fn'])


# ============================================================================
# SOLVER CLASS
# ============================================================================

class PLRCubeSphereAdvection(NumericalSolver):
    """
    PLR Cubed-Sphere Advection Solver.
    
    Features:
    - 2nd-order PLR with MC limiter
    - Cartesian velocity exchange
    - JAX sharding across 6 faces
    """
    
    def __init__(self, N: int, config_file: str = None):
        """
        Initialize solver.
        
        Args:
            N: Grid resolution (N×N per face)
            config_file: Path to YAML config (for parallelization settings)
        """
        self.N = N
        
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
        print(f"  Method: PLR (2nd-order) with MC limiter")
        print(f"  dx: {self.dx:.6f} rad ({self.dx * self.planet.R_sphere / 1e3:.1f} km)")
        print(f"  Planet: {self.planet.name} (R={self.planet.R_sphere/1e6:.3f}×10⁶ m)")
        
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
        """Pre-compute coordinate and metric arrays from geometry module."""
        # Use geometry module (unit sphere, f64) and scale by R² for physical metric
        R = self.planet.R_sphere
        
        # Broadcast coordinates to all faces (same on each face)
        XI1 = jnp.array(self.geometry.XI1)
        XI2 = jnp.array(self.geometry.XI2)
        
        self.XI1_all = jnp.broadcast_to(XI1, (6, self.N, self.N))
        self.XI2_all = jnp.broadcast_to(XI2, (6, self.N, self.N))
        
        # sqrtG from geometry (unit sphere) scaled by R²
        self.sqrtG_all = jnp.array(self.geometry.sqrtG) * R**2
        
        # Apply sharding if enabled
        if self.sharding is not None:
            self.XI1_all = jax.device_put(self.XI1_all, self.sharding)
            self.XI2_all = jax.device_put(self.XI2_all, self.sharding)
            self.sqrtG_all = jax.device_put(self.sqrtG_all, self.sharding)
    
    def initialize(self, test_case='cosine_bell', u0=None, 
                   rotation_period_days=12.0) -> AdvectionState:
        """
        Initialize state using IC modules.
        
        Args:
            test_case: 'cosine_bell' (default)
            u0: Wind speed at equator [m/s]. If None, uses rotation_period_days.
            rotation_period_days: Rotation period [days] (default: 12)
        """
        # Initial condition from IC module
        if test_case == 'cosine_bell':
            q_all = cosine_bell(self.geometry, self.planet, amplitude=1000.0)
            print(f"\nInitial condition: Cosine Bell (Williamson Test 1)")
        else:
            raise ValueError(f"Unknown test case: {test_case}")
        
        # Velocity field from IC module
        Vx_all, Vy_all, Vz_all = solid_body_rotation(
            self.geometry, self.planet, 
            rotation_period_days=rotation_period_days
        )
        
        # Compute u0 for logging
        if u0 is None:
            u0 = 2.0 * jnp.pi * self.planet.R_sphere / (rotation_period_days * 86400.0)
        
        print(f"\nVelocity field: Solid body rotation")
        print(f"  Rotation period: {rotation_period_days} days")
        print(f"  u0 at equator: {float(u0):.2f} m/s")
        
        # Apply sharding if enabled
        if self.sharding is not None:
            q_all = jax.device_put(q_all, self.sharding)
            Vx_all = jax.device_put(Vx_all, self.sharding)
            Vy_all = jax.device_put(Vy_all, self.sharding)
            Vz_all = jax.device_put(Vz_all, self.sharding)
        
        return AdvectionState(q=q_all, Vx=Vx_all, Vy=Vy_all, Vz=Vz_all, 
                             time=0.0, step=0)
    
    def step(self, state: AdvectionState, dt: float) -> AdvectionState:
        """Advance one timestep."""
        return rk3_step_compiled(state, self.dx, dt, self.N,
                                self.XI1_all, self.XI2_all, self.sqrtG_all,
                                self.halo_exchange_fn)
    
    def get_diagnostics(self, state: AdvectionState) -> Dict[str, float]:
        """Compute diagnostics."""
        mass = float(jnp.sum(state.q * self.sqrtG_all * self.dx**2))
        q_max = float(jnp.max(state.q))
        q_min = float(jnp.min(state.q))
        
        # Per-face diagnostics
        face_maxs = [float(jnp.max(state.q[i])) for i in range(6)]
        face_max = int(jnp.argmax(jnp.array(face_maxs)))
        
        return {
            'mass': mass,
            'q_max': q_max,
            'q_min': q_min,
            'face_maxs': face_maxs,
            'face_with_peak': face_max,
        }
    
    # ========================================================================
    # FRAMEWORK INTERFACE METHODS
    # ========================================================================
    
    def get_available_outputs(self) -> Dict[str, List[str]]:
        """Define available output groups and their variables."""
        return {
            'state': ['q', 'Vx', 'Vy', 'Vz'],  # Full fields for viz/restart
            'diagnostics': ['mass', 'q_max', 'q_min', 'face_with_peak']
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
                description="Tracer and velocity fields on all 6 faces"
            ),
            'diagnostics': OutputSpec(
                variables=available['diagnostics'],
                frequency=io_config['diagnostics']['frequency'],
                enabled=io_config['diagnostics']['enabled'],
                description="Mass conservation and peak tracking"
            )
        }
    
    def state_to_output(self, state: AdvectionState, 
                       output_group: str = 'state') -> Dict[str, np.ndarray]:
        """
        Convert state to output format for a specific group.
        
        Args:
            state: Current advection state
            output_group: 'state' or 'diagnostics'
        """
        if output_group == 'state':
            return {
                'q': np.array(state.q),
                'Vx': np.array(state.Vx),
                'Vy': np.array(state.Vy),
                'Vz': np.array(state.Vz),
                'time': state.time,
                'step': state.step,
            }
        elif output_group == 'diagnostics':
            return self.get_diagnostics(state)
        else:
            raise ValueError(f"Unknown output group: {output_group}")
    
    def state_from_checkpoint(self, checkpoint_data: Dict[str, np.ndarray]) -> AdvectionState:
        """
        Restore state from checkpoint data.
        
        Args:
            checkpoint_data: Dictionary containing saved state arrays
        
        Returns:
            Restored advection state
        """
        q = jnp.array(checkpoint_data['q'])
        Vx = jnp.array(checkpoint_data['Vx'])
        Vy = jnp.array(checkpoint_data['Vy'])
        Vz = jnp.array(checkpoint_data['Vz'])
        time = float(checkpoint_data['time'])
        step = int(checkpoint_data['step'])
        
        # Apply sharding if solver was initialized with it
        if hasattr(self, 'sharding') and self.sharding is not None:
            q = jax.device_put(q, self.sharding)
            Vx = jax.device_put(Vx, self.sharding)
            Vy = jax.device_put(Vy, self.sharding)
            Vz = jax.device_put(Vz, self.sharding)
        
        return AdvectionState(q=q, Vx=Vx, Vy=Vy, Vz=Vz, time=time, step=step)


# ============================================================================
# STANDALONE TEST
# ============================================================================

def run_standalone_test(N=60, T_days=2.0, config_file=None):
    """Standalone test for development and validation."""
    print(f"\n{'='*70}")
    print(f"STANDALONE PLR ADVECTION TEST")
    print(f"{'='*70}")
    print(f"Resolution: {N}×{N} per face")
    print(f"Duration: {T_days} days")
    
    # Create solver
    solver = PLRCubeSphereAdvection(N=N, config_file=config_file)
    
    # Initialize
    state = solver.initialize()
    
    # Compute timestep (CFL = 0.5)
    V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
    V_max = float(jnp.max(V_mag))
    target_cfl = 0.5
    dt = target_cfl * (solver.dx * solver.planet.R_sphere) / V_max
    n_steps = int(T_days * 86400 / dt)
    
    print(f"\nTime integration:")
    print(f"  CFL: {target_cfl}")
    print(f"  V_max: {V_max:.2f} m/s")
    print(f"  dt: {dt:.2f} s ({dt/60:.1f} min)")
    print(f"  n_steps: {n_steps}")
    
    # JIT compile
    print(f"\nJIT compiling step function...")
    state = solver.step(state, dt)
    print(f"  ✓ Compilation complete")
    
    # Time loop
    print(f"\nRunning {n_steps} steps...")
    diag0 = solver.get_diagnostics(state)
    mass_initial = diag0['mass']
    
    save_freq = max(1, n_steps // 20)
    
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        if (step + 1) % save_freq == 0:
            diag = solver.get_diagnostics(state)
            mass_err = abs(diag['mass'] - mass_initial) / mass_initial
            t_days = state.time / 86400
            print(f"  Step {step+1:4d} (day {t_days:5.2f}): " +
                  f"q_max={diag['q_max']:7.1f}, " +
                  f"mass_err={mass_err:.2e}, " +
                  f"peak on face {diag['face_with_peak']}")
    
    # Final diagnostics
    diag_final = solver.get_diagnostics(state)
    mass_err = abs(diag_final['mass'] - mass_initial) / mass_initial
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Mass conservation:")
    print(f"  Initial: {mass_initial:.6e}")
    print(f"  Final:   {diag_final['mass']:.6e}")
    print(f"  Error:   {mass_err:.2e}")
    print(f"\nPeak preservation:")
    print(f"  Initial: {diag0['q_max']:.1f}")
    print(f"  Final:   {diag_final['q_max']:.1f}")
    print(f"  Ratio:   {diag_final['q_max']/diag0['q_max']:.3f}")
    print(f"{'='*70}\n")
    
    return solver, state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PLR Cubed-Sphere Advection Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-size', type=int, default=60,
                        help='Grid resolution (N×N cells per face)')
    parser.add_argument('--days', type=float, default=2.0,
                        help='Simulation length in days')
    parser.add_argument('--config', type=str, 
                        default='Config/config_plr_advection.yaml',
                        help='Path to config file (for parallelization)')
    
    args = parser.parse_args()
    
    # Run standalone test
    solver, final_state = run_standalone_test(
        N=args.grid_size,
        T_days=args.days,
        config_file=args.config
    )
    
    print("✓ Test complete!")

