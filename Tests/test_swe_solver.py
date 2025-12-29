"""
Integrated SWE Solver Test: Continuity + Momentum with RK3 Time Integration

Tests the full shallow water equations:
    ∂h/∂t + ∇·(hV) = 0           (continuity)
    ∂V/∂t + ∇B + (ζ+f)k×V = 0   (momentum)

where B = gh + ½|V|² is the Bernoulli function.

For Test Case 2 (steady geostrophic flow):
    - Analytical solution is time-independent
    - After N timesteps, solution should remain close to initial condition
    - This validates: gradients, divergence, vorticity, halo exchange, time integration

Features:
    - 2-deep halo exchanges for 4th-order interior stencils
    - RK3 time integration (Shu-Osher form)
    - CFL-based timestep
    - Benchmark mode

Usage:
    python test_swe_solver.py --precision fp64 --device cpu --N 60 --steps 10
    python test_swe_solver.py --benchmark --N 600 --device gpu --steps 100
"""

import os
import sys
import time
import argparse
from pathlib import Path

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='SWE Solver Integration Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--precision', type=str, default='fp64', 
                        choices=['fp32', 'fp64'])
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['cpu', 'gpu'])
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--N', type=int, default=60,
                        help='Grid resolution')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of timesteps')
    parser.add_argument('--cfl', type=float, default=0.5,
                        help='CFL number for timestep')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Warmup steps for benchmark')
    return parser.parse_args()


# =============================================================================
# JAX CONFIGURATION
# =============================================================================

def configure_jax(precision: str, device: str, num_devices: int):
    if precision == 'fp64':
        os.environ['JAX_ENABLE_X64'] = 'True'
    else:
        os.environ['JAX_ENABLE_X64'] = 'False'
    
    if device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        if num_devices > 1:
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'


args = parse_args()
configure_jax(args.precision, args.device, args.num_devices)

# Now import JAX
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

print(f"\n{'='*70}")
print("JAX CONFIGURATION")
print(f"{'='*70}")
print(f"  Precision: {args.precision} (x64_enabled: {jax.config.x64_enabled})")
print(f"  Device: {args.device}")
print(f"  Devices: {jax.devices()}")
print(f"{'='*70}")

# Add Wraptor to path
WRAPTOR_ROOT = Path(__file__).parent.parent
if "WRAPTOR_ROOT" in os.environ:
    WRAPTOR_ROOT = Path(os.environ["WRAPTOR_ROOT"])
sys.path.insert(0, str(WRAPTOR_ROOT))

from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams
from Solvers.initial_conditions.swe_testcase2 import steady_geostrophic_flow
from Solvers.slope_reconstruction import (
    compute_slopes_fv3,
    compute_jacobian_terms,
    get_jacobian_for_face,
    compute_gram_inverse,
)
from Solvers.halo_exchange import create_communication_schedule
from Solvers.halo_exchange_2deep import (
    make_halo_exchange_2deep,
    extend_to_include_ghosts_2deep,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat):
    """Convert spherical velocity to Cartesian."""
    Vx = -u_lon * jnp.sin(lon) - u_lat * jnp.sin(lat) * jnp.cos(lon)
    Vy =  u_lon * jnp.cos(lon) - u_lat * jnp.sin(lat) * jnp.sin(lon)
    Vz =  u_lat * jnp.cos(lat)
    return Vx, Vy, Vz


# =============================================================================
# SWE RHS COMPUTATION
# =============================================================================

def make_swe_rhs_fn(N: int, dx: float, XI1, XI2, sqrtG_all, lat_all, 
                     halo_exchange_fn, planet: PlanetParams):
    """
    Factory to create JIT-compiled SWE RHS function.
    
    Args:
        N: Grid resolution
        dx: Grid spacing [radians]
        XI1, XI2: (N, N) cubed-sphere coordinates
        sqrtG_all: (6, N, N) metric determinant (unit sphere!)
        lat_all: (6, N, N) latitude
        halo_exchange_fn: JIT-compiled 2-deep halo exchange
        planet: Planet parameters
    
    Returns:
        rhs_fn: (h, Vx, Vy, Vz) -> (dh_dt, dVx_dt, dVy_dt, dVz_dt)
    
    IMPORTANT: Two Jacobian sets needed due to dimensional analysis:
    - Gradient ∇B: uses physical R Jacobians → gives physical m/s² directly
    - Vorticity/Divergence: uses R=1 Jacobians + explicit /R for physical units
    """
    R = planet.R_sphere
    g = planet.gravity
    omega = planet.omega
    
    print("Pre-computing Jacobians for SWE RHS...")
    
    # Physical R Jacobians for gradient (gives correct units directly)
    terms_phys = compute_jacobian_terms(XI1, XI2, R)
    
    # Unit sphere Jacobians for vorticity/divergence (must match unit sqrtG)
    terms_unit = compute_jacobian_terms(XI1, XI2, R=1.0)
    
    jacobians_phys = {}
    jacobians_unit = {}
    for face_id in range(6):
        # Physical Jacobians for gradient
        J11, J12, J21, J22, J31, J32 = get_jacobian_for_face(terms_phys, face_id)
        JTJi_11, JTJi_12, JTJi_22 = compute_gram_inverse(J11, J12, J21, J22, J31, J32)
        jacobians_phys[face_id] = {
            'J11': J11, 'J12': J12, 'J21': J21, 'J22': J22, 'J31': J31, 'J32': J32,
            'JTJi_11': JTJi_11, 'JTJi_12': JTJi_12, 'JTJi_22': JTJi_22
        }
        
        # Unit Jacobians for vorticity/divergence
        J11_u, J12_u, J21_u, J22_u, J31_u, J32_u = get_jacobian_for_face(terms_unit, face_id)
        JTJi_11_u, JTJi_12_u, JTJi_22_u = compute_gram_inverse(J11_u, J12_u, J21_u, J22_u, J31_u, J32_u)
        jacobians_unit[face_id] = {
            'J11': J11_u, 'J12': J12_u, 'J21': J21_u, 'J22': J22_u, 'J31': J31_u, 'J32': J32_u,
            'JTJi_11': JTJi_11_u, 'JTJi_12': JTJi_12_u, 'JTJi_22': JTJi_22_u
        }
    
    # Pre-compute unit vertical vectors k = (X, Y, Z)/R for each face
    k_vectors = {}
    for face_id in range(6):
        # Get Cartesian coordinates on unit sphere
        tan_x1 = jnp.tan(XI1)
        tan_x2 = jnp.tan(XI2)
        delta = jnp.sqrt(1.0 + tan_x1**2 + tan_x2**2)
        
        # Unit sphere coordinates depend on face
        if face_id == 0:
            kx, ky, kz = 1.0/delta, tan_x1/delta, tan_x2/delta
        elif face_id == 1:
            kx, ky, kz = -tan_x1/delta, 1.0/delta, tan_x2/delta
        elif face_id == 2:
            kx, ky, kz = -1.0/delta, -tan_x1/delta, tan_x2/delta
        elif face_id == 3:
            kx, ky, kz = tan_x1/delta, -1.0/delta, tan_x2/delta
        elif face_id == 4:
            kx, ky, kz = tan_x2/delta, tan_x1/delta, 1.0/delta
        elif face_id == 5:
            kx, ky, kz = -tan_x2/delta, tan_x1/delta, -1.0/delta
        
        k_vectors[face_id] = (kx, ky, kz)
    
    print("  ✓ Jacobians and geometry pre-computed")
    
    def compute_rhs_single_face(h, Vx, Vy, Vz, face_id, sqrtG, lat):
        """Compute RHS for a single face (interior only)."""
        J_phys = jacobians_phys[face_id]
        J_unit = jacobians_unit[face_id]
        kx, ky, kz = k_vectors[face_id]
        
        # === BERNOULLI GRADIENT (use physical Jacobians) ===
        B = g * h + 0.5 * (Vx**2 + Vy**2 + Vz**2)
        
        # Slopes in computational coordinates (units: m²/s² per radian)
        dB_dxi1, dB_dxi2 = compute_slopes_fv3(B, dx, N)
        
        # Transform to Cartesian gradient using physical Jacobians
        # This gives physical gradient directly [m/s²]
        grad_xi_1 = J_phys['JTJi_11'] * dB_dxi1 + J_phys['JTJi_12'] * dB_dxi2
        grad_xi_2 = J_phys['JTJi_12'] * dB_dxi1 + J_phys['JTJi_22'] * dB_dxi2
        
        dB_dX = J_phys['J11'] * grad_xi_1 + J_phys['J12'] * grad_xi_2
        dB_dY = J_phys['J21'] * grad_xi_1 + J_phys['J22'] * grad_xi_2
        dB_dZ = J_phys['J31'] * grad_xi_1 + J_phys['J32'] * grad_xi_2
        
        # === VORTICITY TERM (use unit Jacobians + /R) ===
        # Covariant velocity with unit Jacobians
        V1_cov = J_unit['J11'] * Vx + J_unit['J21'] * Vy + J_unit['J31'] * Vz
        V2_cov = J_unit['J12'] * Vx + J_unit['J22'] * Vy + J_unit['J32'] * Vz
        
        # Relative vorticity: ζ = (1/√G)[∂V₂/∂ξ¹ - ∂V₁/∂ξ²] / R
        # Division by R gives physical vorticity [1/s]
        dV2_dxi1, _ = compute_slopes_fv3(V2_cov, dx, N)
        _, dV1_dxi2 = compute_slopes_fv3(V1_cov, dx, N)
        zeta = (dV2_dxi1 - dV1_dxi2) / (sqrtG * R)
        
        # Absolute vorticity
        f = 2.0 * omega * jnp.sin(lat)
        eta = zeta + f
        
        # Cross product k × V
        cross_x = ky * Vz - kz * Vy
        cross_y = kz * Vx - kx * Vz
        cross_z = kx * Vy - ky * Vx
        
        # Vorticity term
        vort_x = eta * cross_x
        vort_y = eta * cross_y
        vort_z = eta * cross_z
        
        # === DIVERGENCE ∇·(hV) (use physical Jacobians like continuity test) ===
        # Covariant velocity with physical Jacobians for divergence
        V1_cov_phys = J_phys['J11'] * Vx + J_phys['J21'] * Vy + J_phys['J31'] * Vz
        V2_cov_phys = J_phys['J12'] * Vx + J_phys['J22'] * Vy + J_phys['J32'] * Vz
        
        # Contravariant velocity
        V1_contra = J_phys['JTJi_11'] * V1_cov_phys + J_phys['JTJi_12'] * V2_cov_phys
        V2_contra = J_phys['JTJi_12'] * V1_cov_phys + J_phys['JTJi_22'] * V2_cov_phys
        
        # Mass flux (weighted by √G) - note sqrtG is unit sphere
        F1 = sqrtG * h * V1_contra
        F2 = sqrtG * h * V2_contra
        
        dF1_dxi1, _ = compute_slopes_fv3(F1, dx, N)
        _, dF2_dxi2 = compute_slopes_fv3(F2, dx, N)
        
        div_hV = (dF1_dxi1 + dF2_dxi2) / sqrtG
        
        # === RHS ===
        dh_dt = -div_hV
        dVx_dt = -dB_dX - vort_x
        dVy_dt = -dB_dY - vort_y
        dVz_dt = -dB_dZ - vort_z
        
        return dh_dt, dVx_dt, dVy_dt, dVz_dt
    
    def swe_rhs(h, Vx, Vy, Vz):
        """
        Compute SWE RHS for all faces with halo exchange.
        
        Args:
            h, Vx, Vy, Vz: (6, N, N) state arrays (interior only)
        
        Returns:
            dh_dt, dVx_dt, dVy_dt, dVz_dt: (6, N, N) tendencies
        """
        # Extend with ghost cells
        h_ghost = extend_to_include_ghosts_2deep(h, N)
        Vx_ghost = extend_to_include_ghosts_2deep(Vx, N)
        Vy_ghost = extend_to_include_ghosts_2deep(Vy, N)
        Vz_ghost = extend_to_include_ghosts_2deep(Vz, N)
        
        # Halo exchange
        h_ghost = halo_exchange_fn(h_ghost)
        Vx_ghost = halo_exchange_fn(Vx_ghost)
        Vy_ghost = halo_exchange_fn(Vy_ghost)
        Vz_ghost = halo_exchange_fn(Vz_ghost)
        
        # Extract interior for RHS computation
        # Note: We use interior values; ghost cells are for boundary stencils
        # The slopes_fv3 uses one-sided stencils at boundaries currently
        # TODO: Update slopes to use ghost cells for centered stencils at boundaries
        
        dh_dt = jnp.zeros((6, N, N))
        dVx_dt = jnp.zeros((6, N, N))
        dVy_dt = jnp.zeros((6, N, N))
        dVz_dt = jnp.zeros((6, N, N))
        
        for face_id in range(6):
            # Extract interior from ghost array
            h_face = h_ghost[face_id, 2:N+2, 2:N+2]
            Vx_face = Vx_ghost[face_id, 2:N+2, 2:N+2]
            Vy_face = Vy_ghost[face_id, 2:N+2, 2:N+2]
            Vz_face = Vz_ghost[face_id, 2:N+2, 2:N+2]
            
            dh, dVx, dVy, dVz = compute_rhs_single_face(
                h_face, Vx_face, Vy_face, Vz_face,
                face_id, sqrtG_all[face_id], lat_all[face_id]
            )
            
            dh_dt = dh_dt.at[face_id].set(dh)
            dVx_dt = dVx_dt.at[face_id].set(dVx)
            dVy_dt = dVy_dt.at[face_id].set(dVy)
            dVz_dt = dVz_dt.at[face_id].set(dVz)
        
        return dh_dt, dVx_dt, dVy_dt, dVz_dt
    
    rhs_fn = jax.jit(swe_rhs)
    print("  ✓ SWE RHS function JIT compiled")
    
    return rhs_fn


# =============================================================================
# RK3 TIME INTEGRATION
# =============================================================================

def make_rk3_step_fn(rhs_fn):
    """
    Create RK3 (Shu-Osher) time stepping function.
    
    Shu-Osher form:
        u^(1) = u^n + dt * L(u^n)
        u^(2) = 3/4 u^n + 1/4 u^(1) + 1/4 dt * L(u^(1))
        u^(n+1) = 1/3 u^n + 2/3 u^(2) + 2/3 dt * L(u^(2))
    """
    def rk3_step(h, Vx, Vy, Vz, dt):
        # Stage 1
        dh1, dVx1, dVy1, dVz1 = rhs_fn(h, Vx, Vy, Vz)
        h1 = h + dt * dh1
        Vx1 = Vx + dt * dVx1
        Vy1 = Vy + dt * dVy1
        Vz1 = Vz + dt * dVz1
        
        # Stage 2
        dh2, dVx2, dVy2, dVz2 = rhs_fn(h1, Vx1, Vy1, Vz1)
        h2 = 0.75*h + 0.25*h1 + 0.25*dt*dh2
        Vx2 = 0.75*Vx + 0.25*Vx1 + 0.25*dt*dVx2
        Vy2 = 0.75*Vy + 0.25*Vy1 + 0.25*dt*dVy2
        Vz2 = 0.75*Vz + 0.25*Vz1 + 0.25*dt*dVz2
        
        # Stage 3
        dh3, dVx3, dVy3, dVz3 = rhs_fn(h2, Vx2, Vy2, Vz2)
        h_new = (1.0/3.0)*h + (2.0/3.0)*h2 + (2.0/3.0)*dt*dh3
        Vx_new = (1.0/3.0)*Vx + (2.0/3.0)*Vx2 + (2.0/3.0)*dt*dVx3
        Vy_new = (1.0/3.0)*Vy + (2.0/3.0)*Vy2 + (2.0/3.0)*dt*dVy3
        Vz_new = (1.0/3.0)*Vz + (2.0/3.0)*Vz2 + (2.0/3.0)*dt*dVz3
        
        return h_new, Vx_new, Vy_new, Vz_new
    
    return jax.jit(rk3_step)


# =============================================================================
# MAIN TEST
# =============================================================================

def run_swe_test(N: int, steps: int, cfl: float, benchmark: bool = False, warmup: int = 2):
    """
    Run SWE solver test.
    
    For Test Case 2, the analytical solution is steady.
    After `steps` timesteps, we measure drift from initial condition.
    """
    print(f"\n{'='*70}")
    print(f"SWE SOLVER TEST")
    print(f"  N={N}, Steps={steps}, CFL={cfl}")
    print(f"  Precision: {args.precision}, Device: {args.device}")
    print(f"{'='*70}")
    
    # ========================================================================
    # SETUP
    # ========================================================================
    t_setup_start = time.perf_counter()
    
    geom = CubedSphereGeometry.create(N)
    planet = PlanetParams()
    
    R = planet.R_sphere
    g = planet.gravity
    dx = geom.dx
    
    h0, u0 = 8000.0, 40.0
    
    # Initial conditions
    h_init, u_lon, u_lat = steady_geostrophic_flow(geom, planet, h0=h0, u0=u0)
    h_init = jnp.array(h_init)
    u_lon = jnp.array(u_lon)
    u_lat = jnp.array(u_lat)
    
    lat, lon = geom.get_lat_lon_all_faces()
    lat = jnp.array(lat)
    lon = jnp.array(lon)
    
    XI1 = jnp.array(geom.XI1)
    XI2 = jnp.array(geom.XI2)
    sqrtG = jnp.array(geom.sqrtG)
    
    # Convert velocity to Cartesian
    Vx_init = jnp.zeros((6, N, N))
    Vy_init = jnp.zeros((6, N, N))
    Vz_init = jnp.zeros((6, N, N))
    
    for face in range(6):
        Vx, Vy, Vz = spherical_to_cartesian_velocity(
            u_lon[face], u_lat[face], lon[face], lat[face]
        )
        Vx_init = Vx_init.at[face].set(Vx)
        Vy_init = Vy_init.at[face].set(Vy)
        Vz_init = Vz_init.at[face].set(Vz)
    
    # Timestep from CFL
    # dt = CFL * dx * R / max_speed
    max_speed = jnp.sqrt(g * h0) + u0  # gravity wave + advection
    dt = cfl * dx * R / max_speed
    
    print(f"\n  Grid: 6 × {N} × {N} = {6*N*N:,} cells")
    
    # Physical grid spacing (angular dx converted to km)
    dx_km = dx * R / 1000  # km at equator
    print(f"  Grid spacing:")
    print(f"    dx = {dx:.6f} rad = {dx_km:.1f} km (equatorial)")
    print(f"    dx varies: {dx_km:.1f} - {dx_km * 1.4:.1f} km (face center to corner)")
    
    # Wave speeds for CFL
    c = jnp.sqrt(g * h0)  # Gravity wave speed
    print(f"  Wave speeds:")
    print(f"    Gravity wave c = √(gh₀) = {float(c):.1f} m/s")
    print(f"    Flow velocity u₀ = {u0:.1f} m/s") 
    print(f"    CFL uses c + u₀ = {float(c + u0):.1f} m/s")
    
    print(f"  Timestep:")
    print(f"    dt = {dt:.1f} s = {dt/60:.1f} min (CFL={cfl})")
    print(f"    Total integration: {steps * dt:.0f} s = {steps * dt / 3600:.2f} hours")
    
    # Create halo exchange function
    schedule = create_communication_schedule()
    halo_exchange_fn = make_halo_exchange_2deep(schedule, N)
    
    # Create RHS and RK3 functions
    rhs_fn = make_swe_rhs_fn(N, float(dx), XI1, XI2, sqrtG, lat, halo_exchange_fn, planet)
    rk3_step = make_rk3_step_fn(rhs_fn)
    
    t_setup = time.perf_counter() - t_setup_start
    print(f"\n  Setup time: {t_setup*1000:.1f} ms")
    
    # ========================================================================
    # WARMUP (JIT compile)
    # ========================================================================
    print(f"\n  Warming up ({warmup} steps)...")
    t_warmup_start = time.perf_counter()
    
    h, Vx, Vy, Vz = h_init, Vx_init, Vy_init, Vz_init
    for i in range(warmup):
        h, Vx, Vy, Vz = rk3_step(h, Vx, Vy, Vz, dt)
        jax.block_until_ready(h)
    
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"  Warmup complete: {t_warmup*1000:.1f} ms ({t_warmup/warmup*1000:.1f} ms/step)")
    
    # ========================================================================
    # TIMED INTEGRATION
    # ========================================================================
    print(f"\n  Running {steps} timesteps...")
    
    # Reset to initial condition
    h, Vx, Vy, Vz = h_init, Vx_init, Vy_init, Vz_init
    
    t_run_start = time.perf_counter()
    
    step_times = []
    for i in range(steps):
        t0 = time.perf_counter()
        h, Vx, Vy, Vz = rk3_step(h, Vx, Vy, Vz, dt)
        jax.block_until_ready(h)
        step_times.append(time.perf_counter() - t0)
        
        # Check for NaN
        if jnp.any(jnp.isnan(h)):
            print(f"\n  *** NaN detected at step {i+1}! ***")
            break
    
    t_run = time.perf_counter() - t_run_start
    step_times = np.array(step_times) * 1000  # ms
    
    # ========================================================================
    # ERROR ANALYSIS
    # ========================================================================
    # For Test Case 2, analytical solution is steady
    # Error = deviation from initial condition
    
    h_error = h - h_init
    Vx_error = Vx - Vx_init
    Vy_error = Vy - Vy_init
    Vz_error = Vz - Vz_init
    V_error_mag = jnp.sqrt(Vx_error**2 + Vy_error**2 + Vz_error**2)
    V_init_mag = jnp.sqrt(Vx_init**2 + Vy_init**2 + Vz_init**2)
    
    # Relative errors
    h_rel_L2 = float(jnp.sqrt(jnp.mean(h_error**2)) / h0)
    h_rel_Linf = float(jnp.max(jnp.abs(h_error)) / h0)
    
    V_rel_L2 = float(jnp.sqrt(jnp.mean(V_error_mag**2)) / u0)
    V_rel_Linf = float(jnp.max(V_error_mag) / u0)
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\n  Timing:")
    print(f"    Total time:      {t_run*1000:.1f} ms")
    print(f"    Time/step:       {step_times.mean():.2f} ± {step_times.std():.2f} ms")
    print(f"    Min/Max:         {step_times.min():.2f} / {step_times.max():.2f} ms")
    
    total_cells = 6 * N * N
    throughput = total_cells / (step_times.mean() / 1000)
    print(f"\n  Throughput:")
    print(f"    {throughput:.2e} cells/second")
    print(f"    {throughput * 3:.2e} cell-RHS-evals/second (RK3)")
    
    print(f"\n  Error (drift from initial condition after {steps} steps):")
    print(f"    Height h:")
    print(f"      L2 relative:   {h_rel_L2:.2e}")
    print(f"      Linf relative: {h_rel_Linf:.2e}")
    print(f"    Velocity V:")
    print(f"      L2 relative:   {V_rel_L2:.2e}")
    print(f"      Linf relative: {V_rel_Linf:.2e}")
    
    # Stability check
    stable = not jnp.any(jnp.isnan(h)) and h_rel_Linf < 1.0
    print(f"\n  Stability: {'STABLE' if stable else 'UNSTABLE'}")
    
    return {
        'N': N,
        'steps': steps,
        'dt': float(dt),
        'cfl': cfl,
        'time_per_step_ms': float(step_times.mean()),
        'throughput': throughput,
        'h_rel_L2': h_rel_L2,
        'h_rel_Linf': h_rel_Linf,
        'V_rel_L2': V_rel_L2,
        'V_rel_Linf': V_rel_Linf,
        'stable': stable,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    result = run_swe_test(
        N=args.N,
        steps=args.steps,
        cfl=args.cfl,
        benchmark=args.benchmark,
        warmup=args.warmup
    )
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}\n")
