"""
Unit Test: Continuity Equation using FV3 divergence operator

Tests the mass conservation equation:
    ∂h/∂t + ∇·(hV) = 0

For Test Case 2 (steady geostrophic flow):
    ∂h/∂t = 0  →  ∇·(hV) = 0

The divergence in curvilinear coordinates:
    ∇·(hV) = (1/√G) [∂(√G h V¹)/∂ξ¹ + ∂(√G h V²)/∂ξ²]

Where V¹, V² are contravariant velocity components.

Features:
    - fp32/fp64 precision toggle
    - CPU/GPU device selection
    - L1/L2/Linf error norms
    - Benchmark mode with warmup

Usage:
    python test_continuity_module.py --precision fp64 --device cpu
    python test_continuity_module.py --precision fp32 --device gpu --N 120
    python test_continuity_module.py --benchmark --N 600 --device gpu
"""

import os
import sys
import time
import argparse
from pathlib import Path

# =============================================================================
# ARGUMENT PARSING - Must happen before JAX configuration
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Continuity Equation Unit Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--precision', type=str, default='fp64', 
                        choices=['fp32', 'fp64'],
                        help='Floating point precision')
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['cpu', 'gpu'],
                        help='Device type')
    parser.add_argument('--num_devices', type=int, default=1,
                        help='Number of devices (1-6 for CPU cores, or GPU count)')
    parser.add_argument('--N', type=int, default=None,
                        help='Single resolution (overrides --resolutions)')
    parser.add_argument('--resolutions', type=str, default='15,30,60',
                        help='Comma-separated resolutions for convergence study')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode with warmup and averaged timing')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of timed iterations for benchmark mode')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations for benchmark mode')
    return parser.parse_args()


# =============================================================================
# JAX CONFIGURATION
# =============================================================================

def configure_jax(precision: str, device: str, num_devices: int):
    """Configure JAX precision and device BEFORE importing JAX."""
    if precision == 'fp64':
        os.environ['JAX_ENABLE_X64'] = 'True'
    else:
        os.environ['JAX_ENABLE_X64'] = 'False'
    
    if device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        if num_devices > 1:
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'


# Configure before importing JAX
args = parse_args()
configure_jax(args.precision, args.device, args.num_devices)

# =============================================================================
# NOW import JAX and other modules
# =============================================================================

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

# Print configuration
print(f"\n{'='*70}")
print("JAX CONFIGURATION")
print(f"{'='*70}")
print(f"  Precision: {args.precision} (x64_enabled: {jax.config.x64_enabled})")
print(f"  Device type: {args.device}")
print(f"  Num devices: {args.num_devices}")
print(f"  Available devices: {jax.devices()}")
print(f"  Default backend: {jax.default_backend()}")
dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
print(f"  Working dtype: {dtype}")

# Setup sharding if num_devices > 1
if args.num_devices > 1:
    devices = jax.devices()[:args.num_devices]
    mesh = Mesh(np.array(devices), ('faces',))
    sharding = NamedSharding(mesh, P('faces'))
    print(f"  Sharding: {args.num_devices} devices, 1 face per device")
else:
    mesh = None
    sharding = None
    print(f"  Sharding: disabled (single device)")

print(f"{'='*70}")

# Add Wraptor root to path
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat):
    """Convert spherical velocity (u_lon, u_lat) to Cartesian (Vx, Vy, Vz)."""
    Vx = -u_lon * jnp.sin(lon) - u_lat * jnp.sin(lat) * jnp.cos(lon)
    Vy =  u_lon * jnp.cos(lon) - u_lat * jnp.sin(lat) * jnp.sin(lon)
    Vz =  u_lat * jnp.cos(lat)
    return Vx, Vy, Vz


def compute_contravariant_velocity(Vx, Vy, Vz, J11, J12, J21, J22, J31, J32,
                                    JTJi_11, JTJi_12, JTJi_22):
    """
    Convert Cartesian velocity to contravariant components.
    
    V^i = g^{ij} V_j  where g^{ij} = (J^T J)^{-1}
    
    First get covariant: V_i = J_{ki} V^k
    Then contravariant: V^i = g^{ij} V_j
    
    Args:
        Vx, Vy, Vz: Cartesian velocity components [m/s]
        J11...J32: Jacobian components (∂x/∂ξ)
        JTJi_11, JTJi_12, JTJi_22: Inverse Gram matrix (J^T J)^{-1}
    
    Returns:
        V1_contra, V2_contra: Contravariant velocity components
    """
    # Covariant components: V_i = e_i · V where e_i = J_{:,i}
    V1_cov = J11 * Vx + J21 * Vy + J31 * Vz
    V2_cov = J12 * Vx + J22 * Vy + J32 * Vz
    
    # Contravariant: V^i = g^{ij} V_j
    V1_contra = JTJi_11 * V1_cov + JTJi_12 * V2_cov
    V2_contra = JTJi_12 * V1_cov + JTJi_22 * V2_cov
    
    return V1_contra, V2_contra


def compute_divergence_single_face(h, V1_contra, V2_contra, sqrtG, dx, N):
    """
    Compute divergence of mass flux for a single face.
    
    ∇·(hV) = (1/√G) [∂(√G h V¹)/∂ξ¹ + ∂(√G h V²)/∂ξ²]
    
    Args:
        h: (N, N) height field [m]
        V1_contra, V2_contra: (N, N) contravariant velocity
        sqrtG: (N, N) metric determinant
        dx: Grid spacing [radians]
        N: Grid size
    
    Returns:
        div_hV: (N, N) divergence of mass flux [m/s]
    """
    # Mass flux components (weighted by √G)
    F1 = sqrtG * h * V1_contra
    F2 = sqrtG * h * V2_contra
    
    # Derivatives using FV3 stencils
    dF1_dxi1, _ = compute_slopes_fv3(F1, dx, N)
    _, dF2_dxi2 = compute_slopes_fv3(F2, dx, N)
    
    # Divergence
    div_hV = (dF1_dxi1 + dF2_dxi2) / sqrtG
    
    return div_hV


# =============================================================================
# DIVERGENCE FACTORY
# =============================================================================

def make_divergence_fn(N: int, dx: float, XI1: jnp.ndarray, XI2: jnp.ndarray, 
                        sqrtG_all: jnp.ndarray, R: float):
    """
    Factory creates JIT-compiled divergence function with pre-computed geometry.
    
    Args:
        N: Grid size
        dx: Grid spacing [radians]
        XI1: (N, N) ξ¹ coordinates
        XI2: (N, N) ξ² coordinates
        sqrtG_all: (6, N, N) metric determinant for all faces
        R: Planet radius [m]
    
    Returns:
        divergence_fn: JIT-compiled function 
                       (h_all, Vx_all, Vy_all, Vz_all) -> div_hV_all
    """
    print("Pre-computing Jacobians for FV3 divergence...")
    
    # Pre-compute base terms
    terms = compute_jacobian_terms(XI1, XI2, R)
    
    # Pre-compute per-face Jacobians and Gram inverses
    jacobians = {}
    for face_id in range(6):
        J11, J12, J21, J22, J31, J32 = get_jacobian_for_face(terms, face_id)
        JTJi_11, JTJi_12, JTJi_22 = compute_gram_inverse(J11, J12, J21, J22, J31, J32)
        
        jacobians[face_id] = {
            'J11': J11, 'J12': J12, 'J21': J21, 'J22': J22, 'J31': J31, 'J32': J32,
            'JTJi_11': JTJi_11, 'JTJi_12': JTJi_12, 'JTJi_22': JTJi_22
        }
    
    print(f"  ✓ Jacobians pre-computed for all 6 faces")
    
    def divergence_all_faces(h_all, Vx_all, Vy_all, Vz_all):
        """
        Compute divergence of mass flux on all 6 faces.
        
        Args:
            h_all: (6, N, N) height field
            Vx_all, Vy_all, Vz_all: (6, N, N) Cartesian velocity
        
        Returns:
            div_hV_all: (6, N, N) divergence of mass flux
        """
        div_hV_all = jnp.zeros_like(h_all)
        
        for face_id in range(6):
            J = jacobians[face_id]
            
            # Get contravariant velocity
            V1_contra, V2_contra = compute_contravariant_velocity(
                Vx_all[face_id], Vy_all[face_id], Vz_all[face_id],
                J['J11'], J['J12'], J['J21'], J['J22'], J['J31'], J['J32'],
                J['JTJi_11'], J['JTJi_12'], J['JTJi_22']
            )
            
            # Compute divergence
            div_hV = compute_divergence_single_face(
                h_all[face_id], V1_contra, V2_contra,
                sqrtG_all[face_id], dx, N
            )
            
            div_hV_all = div_hV_all.at[face_id].set(div_hV)
        
        return div_hV_all
    
    divergence_fn_jit = jax.jit(divergence_all_faces)
    
    print(f"  ✓ Divergence function JIT compiled")
    print(f"    Input: h(6,{N},{N}), V(6,{N},{N})×3")
    print(f"    Output: ∇·(hV) (6,{N},{N})")
    
    return divergence_fn_jit


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def test_continuity(N: int = 30, verbose: bool = True, sharding=None):
    """
    Test continuity equation using FV3 divergence operator.
    
    For Test Case 2 (steady geostrophic flow):
        ∇·(hV) should be zero
    
    Args:
        N: Grid resolution
        verbose: Print detailed output
        sharding: Optional JAX sharding
    
    Returns dict with test results.
    """
    t_wall_start = time.perf_counter()
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"CONTINUITY EQUATION TEST (N={N})")
        print("=" * 70)
    
    # ========================================================================
    # SETUP
    # ========================================================================
    t_setup_start = time.perf_counter()
    
    geom = CubedSphereGeometry.create(N)
    planet = PlanetParams()
    
    R = planet.R_sphere
    g = planet.gravity
    omega = planet.omega
    dx = geom.dx
    
    h0, u0 = 8000.0, 40.0
    
    if verbose:
        print(f"\n  dx = {dx:.6f} rad = {np.degrees(dx):.3f}°")
        print(f"  dx² = {dx**2:.2e}")
        print(f"  h0 = {h0} m, u0 = {u0} m/s")
        if sharding is not None:
            print(f"  Sharding: enabled ({args.num_devices} devices)")
    
    # Initial conditions
    h, u_lon, u_lat = steady_geostrophic_flow(geom, planet, h0=h0, u0=u0)
    h = jnp.array(h)
    u_lon = jnp.array(u_lon)
    u_lat = jnp.array(u_lat)
    
    lat, lon = geom.get_lat_lon_all_faces()
    lat = jnp.array(lat)
    lon = jnp.array(lon)
    
    XI1 = jnp.array(geom.XI1)
    XI2 = jnp.array(geom.XI2)
    
    # sqrtG from geometry (unit sphere, need to scale by R² for physical)
    # Actually for divergence we use unit-sphere sqrtG since our dx is in radians
    sqrtG_all = jnp.array(geom.sqrtG)
    
    # Convert velocity to Cartesian
    Vx_all = jnp.zeros((6, N, N))
    Vy_all = jnp.zeros((6, N, N))
    Vz_all = jnp.zeros((6, N, N))
    
    for face in range(6):
        Vx, Vy, Vz = spherical_to_cartesian_velocity(
            u_lon[face], u_lat[face], lon[face], lat[face]
        )
        Vx_all = Vx_all.at[face].set(Vx)
        Vy_all = Vy_all.at[face].set(Vy)
        Vz_all = Vz_all.at[face].set(Vz)
    
    # Apply sharding
    if sharding is not None:
        h = jax.device_put(h, sharding)
        Vx_all = jax.device_put(Vx_all, sharding)
        Vy_all = jax.device_put(Vy_all, sharding)
        Vz_all = jax.device_put(Vz_all, sharding)
        sqrtG_all = jax.device_put(sqrtG_all, sharding)
    
    t_setup = time.perf_counter() - t_setup_start
    
    # ========================================================================
    # DIVERGENCE COMPUTATION
    # ========================================================================
    t_div_start = time.perf_counter()
    
    divergence_fn = make_divergence_fn(N, float(dx), XI1, XI2, sqrtG_all, R)
    
    # Compute divergence
    div_hV = divergence_fn(h, Vx_all, Vy_all, Vz_all)
    jax.block_until_ready(div_hV)
    
    t_div = time.perf_counter() - t_div_start
    
    # ========================================================================
    # ERROR ANALYSIS
    # ========================================================================
    # For steady state, analytical divergence is exactly zero
    # We measure error relative to a characteristic scale: h0 * u0 / R
    # This is the typical magnitude of hV/R which has units [m/s]
    
    # Actually ∇·(hV) has units of [m/s] (height flux per radian)
    # Scale by characteristic: h0 * u0 / R ≈ 8000 * 40 / 6.371e6 ≈ 0.05 m/s
    char_scale = h0 * u0 / R
    
    # Compute error norms (excluding near-pole points where flow is weak)
    errors = []
    for face in range(6):
        lat_f = lat[face]
        # Mask: exclude poles and equator where velocity/flux is small
        mask = (jnp.abs(lat_f) > jnp.radians(10)) & (jnp.abs(lat_f) < jnp.radians(85))
        
        # Relative error normalized by characteristic scale
        rel_error = jnp.abs(div_hV[face]) / char_scale
        valid = rel_error[mask]
        errors.extend(valid.tolist())
    
    errors = np.array(errors)
    L1 = np.mean(np.abs(errors))
    L2 = np.sqrt(np.mean(errors**2))
    Linf = np.max(np.abs(errors))
    
    # Also compute absolute errors for reference
    abs_errors = []
    for face in range(6):
        lat_f = lat[face]
        mask = (jnp.abs(lat_f) > jnp.radians(10)) & (jnp.abs(lat_f) < jnp.radians(85))
        valid = jnp.abs(div_hV[face])[mask]
        abs_errors.extend(valid.tolist())
    
    abs_errors = np.array(abs_errors)
    L1_abs = np.mean(abs_errors)
    L2_abs = np.sqrt(np.mean(abs_errors**2))
    Linf_abs = np.max(abs_errors)
    
    # ========================================================================
    # TIMING SUMMARY
    # ========================================================================
    t_wall = time.perf_counter() - t_wall_start
    
    if verbose:
        print(f"\n  TIMING (wall clock):")
        print(f"    Setup:       {t_setup*1000:8.1f} ms")
        print(f"    Divergence:  {t_div*1000:8.1f} ms")
        print(f"    Total:       {t_wall*1000:8.1f} ms")
        
        print(f"\n  DIVERGENCE ∇·(hV) (should be 0 for steady state):")
        print(f"    Characteristic scale: h₀u₀/R = {char_scale:.2e} m/s")
        
        print(f"\n  RELATIVE ERROR NORMS (normalized by h₀u₀/R):")
        print(f"    L1:   {L1:.2e}")
        print(f"    L2:   {L2:.2e}")
        print(f"    Linf: {Linf:.2e}")
        
        print(f"\n  ABSOLUTE ERROR NORMS [m/s]:")
        print(f"    L1:   {L1_abs:.2e}")
        print(f"    L2:   {L2_abs:.2e}")
        print(f"    Linf: {Linf_abs:.2e}")
        
        print(f"\n    Expected truncation ~ dx² = {dx**2:.2e}")
    
    return {
        'N': N,
        'dx': float(dx),
        'L1': L1, 'L2': L2, 'Linf': Linf,
        'L1_abs': L1_abs, 'L2_abs': L2_abs, 'Linf_abs': Linf_abs,
        'char_scale': char_scale,
        'expected_truncation': dx**2,
        't_setup_ms': t_setup * 1000,
        't_div_ms': t_div * 1000,
        't_wall_ms': t_wall * 1000,
    }


# =============================================================================
# CONVERGENCE STUDY
# =============================================================================

def run_convergence_study(resolutions: list, sharding=None):
    """Run convergence study across resolutions."""
    print(f"\n{'='*70}")
    print("CONVERGENCE STUDY: CONTINUITY EQUATION")
    print(f"  Precision: {args.precision}, Device: {args.device}, Num devices: {args.num_devices}")
    print(f"{'='*70}")
    
    results = []
    for N in resolutions:
        r = test_continuity(N, sharding=sharding)
        results.append(r)
    
    # Summary table
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  Relative Error Norms (normalized by h₀u₀/R):")
    print(f"    {'N':>6} {'dx':>10} {'L1':>12} {'L2':>12} {'Linf':>12} {'dx²':>12}")
    print(f"    {'-'*66}")
    
    for r in results:
        print(f"    {r['N']:>6} {r['dx']:>10.4f} {r['L1']:>12.2e} {r['L2']:>12.2e} {r['Linf']:>12.2e} {r['expected_truncation']:>12.2e}")
    
    # Timing summary
    print(f"\n  Timing (wall clock, ms):")
    print(f"    {'N':>6} {'Setup':>10} {'Divergence':>10} {'Total':>10}")
    print(f"    {'-'*40}")
    
    for r in results:
        print(f"    {r['N']:>6} {r['t_setup_ms']:>10.0f} {r['t_div_ms']:>10.0f} {r['t_wall_ms']:>10.0f}")
    
    # Convergence rates
    if len(results) >= 2:
        r1, r2 = results[-2], results[-1]
        
        rate_L1 = np.log(r1['L1'] / r2['L1']) / np.log(r1['dx'] / r2['dx'])
        rate_L2 = np.log(r1['L2'] / r2['L2']) / np.log(r1['dx'] / r2['dx'])
        rate_Linf = np.log(r1['Linf'] / r2['Linf']) / np.log(r1['dx'] / r2['dx'])
        
        print(f"\n  Convergence rates (N={r1['N']}→{r2['N']}):")
        print(f"    L1: {rate_L1:.2f}   L2: {rate_L2:.2f}   Linf: {rate_Linf:.2f}")
        print(f"    Expected: 2.0")
        
        passed = rate_L2 > 1.8
        print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    
    return results


# =============================================================================
# BENCHMARK MODE
# =============================================================================

def run_benchmark(N: int, iterations: int = 10, warmup: int = 2, sharding=None):
    """Run benchmark with warmup and averaged timing."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK MODE: CONTINUITY EQUATION")
    print(f"  N={N}, Warmup={warmup}, Iterations={iterations}")
    print(f"  Precision: {args.precision}, Device: {args.device}")
    print(f"{'='*70}")
    
    total_cells = 6 * N * N
    print(f"\n  Grid: 6 × {N} × {N} = {total_cells:,} cells")
    
    # Setup
    print(f"\n  Setting up...")
    t_setup_start = time.perf_counter()
    
    geom = CubedSphereGeometry.create(N)
    planet = PlanetParams()
    
    R = planet.R_sphere
    dx = geom.dx
    h0, u0 = 8000.0, 40.0
    
    h, u_lon, u_lat = steady_geostrophic_flow(geom, planet, h0=h0, u0=u0)
    h = jnp.array(h)
    u_lon = jnp.array(u_lon)
    u_lat = jnp.array(u_lat)
    
    lat, lon = geom.get_lat_lon_all_faces()
    lat = jnp.array(lat)
    lon = jnp.array(lon)
    
    XI1 = jnp.array(geom.XI1)
    XI2 = jnp.array(geom.XI2)
    sqrtG_all = jnp.array(geom.sqrtG)
    
    Vx_all = jnp.zeros((6, N, N))
    Vy_all = jnp.zeros((6, N, N))
    Vz_all = jnp.zeros((6, N, N))
    
    for face in range(6):
        Vx, Vy, Vz = spherical_to_cartesian_velocity(
            u_lon[face], u_lat[face], lon[face], lat[face]
        )
        Vx_all = Vx_all.at[face].set(Vx)
        Vy_all = Vy_all.at[face].set(Vy)
        Vz_all = Vz_all.at[face].set(Vz)
    
    if sharding is not None:
        h = jax.device_put(h, sharding)
        Vx_all = jax.device_put(Vx_all, sharding)
        Vy_all = jax.device_put(Vy_all, sharding)
        Vz_all = jax.device_put(Vz_all, sharding)
        sqrtG_all = jax.device_put(sqrtG_all, sharding)
    
    t_setup = time.perf_counter() - t_setup_start
    print(f"  Setup complete: {t_setup*1000:.1f} ms")
    
    # Create JIT-compiled function
    print(f"\n  Creating JIT-compiled divergence function...")
    divergence_fn = make_divergence_fn(N, float(dx), XI1, XI2, sqrtG_all, R)
    
    # Warmup
    print(f"\n  Warming up ({warmup} iterations)...")
    t_warmup_start = time.perf_counter()
    
    for i in range(warmup):
        div_hV = divergence_fn(h, Vx_all, Vy_all, Vz_all)
        jax.block_until_ready(div_hV)
    
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"  Warmup complete: {t_warmup*1000:.1f} ms ({t_warmup/warmup*1000:.1f} ms/iter)")
    
    # Timed iterations
    print(f"\n  Running {iterations} timed iterations...")
    
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        div_hV = divergence_fn(h, Vx_all, Vy_all, Vz_all)
        jax.block_until_ready(div_hV)
        times.append(time.perf_counter() - t0)
    
    times = np.array(times) * 1000  # ms
    
    # Results
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    
    print(f"\n  Timing (ms) over {iterations} iterations:")
    print(f"    {'Operation':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"    {'-'*60}")
    print(f"    {'Divergence':<20} {times.mean():>10.2f} {times.std():>10.2f} {times.min():>10.2f} {times.max():>10.2f}")
    
    throughput = total_cells / (times.mean() / 1000)
    
    print(f"\n  Throughput:")
    print(f"    {throughput:.2e} cells/second")
    
    print(f"\n  JIT Compilation Overhead:")
    print(f"    First call (warmup): {t_warmup/warmup*1000:.1f} ms")
    print(f"    After warmup:        {times.mean():.1f} ms")
    print(f"    Speedup:             {(t_warmup/warmup*1000) / times.mean():.1f}x")
    
    return {
        'N': N,
        'total_cells': total_cells,
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'throughput': throughput,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if args.benchmark:
        if args.N is None:
            print("ERROR: --benchmark requires --N to specify resolution")
            sys.exit(1)
        run_benchmark(args.N, iterations=args.iterations, warmup=args.warmup, sharding=sharding)
    elif args.N is not None:
        test_continuity(args.N, sharding=sharding)
    else:
        resolutions = [int(n) for n in args.resolutions.split(',')]
        run_convergence_study(resolutions, sharding=sharding)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}\n")
