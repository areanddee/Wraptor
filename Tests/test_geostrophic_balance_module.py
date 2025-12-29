"""
Unit Test: Geostrophic Balance using slope_reconstruction module

Tests three things:
1. |∇B|_numerical vs |(ζ+f)k×V|_analytical  (validates gradient accuracy)
2. |∇B + (ζ+f)k×V|_numerical  (validates momentum RHS residual)
3. η_numerical vs η_analytical  (validates vorticity computation)

Features:
    - fp32/fp64 precision toggle
    - CPU/GPU device selection (following fv_plr_cubesphere_adv.py pattern)
    - Performance timing

Usage:
    python test_geostrophic_balance_module.py --precision fp64 --device cpu
    python test_geostrophic_balance_module.py --precision fp32 --device gpu --N 120
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
        description='Geostrophic Balance Unit Test',
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
    return parser.parse_args()


# =============================================================================
# JAX CONFIGURATION - Following fv_plr_cubesphere_adv.py pattern
# =============================================================================

def configure_jax(precision: str, device: str, num_devices: int):
    """
    Configure JAX precision and device BEFORE importing JAX.
    
    Pattern from fv_plr_cubesphere_adv.py setup_sharding().
    
    For multi-core CPU: sets XLA_FLAGS to create virtual devices.
    """
    # Precision
    if precision == 'fp64':
        os.environ['JAX_ENABLE_X64'] = 'True'
    else:
        os.environ['JAX_ENABLE_X64'] = 'False'
    
    # Device configuration
    if device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        # Multi-core CPU: create virtual devices (PLR pattern)
        if num_devices > 1:
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'
    # For GPU: don't set JAX_PLATFORM_NAME, let JAX auto-detect


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

# Print configuration (following PLR pattern)
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
    make_gradient_fn, 
    compute_slopes_fv3,
    compute_jacobian_terms,
    get_jacobian_for_face
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


def compute_vorticity_term_cartesian(eta, Vx, Vy, Vz, X, Y, Z, R):
    """Compute (ζ+f)k×V in Cartesian coordinates."""
    kx, ky, kz = X/R, Y/R, Z/R
    cross_x = ky * Vz - kz * Vy
    cross_y = kz * Vx - kx * Vz
    cross_z = kx * Vy - ky * Vx
    return eta * cross_x, eta * cross_y, eta * cross_z


def compute_absolute_vorticity_numerical(
    Vx, Vy, Vz, XI1, XI2, face_id, sqrtG, lat, omega, dx, N, R
):
    """
    Compute absolute vorticity η = ζ + f numerically.
    
    ζ = (1/√G)[∂V₂/∂ξ¹ - ∂V₁/∂ξ²] / R
    """
    # Unit sphere Jacobian to match geometry sqrtG
    jacobian_terms_unit = compute_jacobian_terms(XI1, XI2, R=1.0)
    J11, J12, J21, J22, J31, J32 = get_jacobian_for_face(jacobian_terms_unit, face_id)
    
    # Covariant components
    V1_cov = J11 * Vx + J21 * Vy + J31 * Vz
    V2_cov = J12 * Vx + J22 * Vy + J32 * Vz
    
    # FV3 stencil derivatives
    dV2_dxi1, _ = compute_slopes_fv3(V2_cov, dx, N)
    _, dV1_dxi2 = compute_slopes_fv3(V1_cov, dx, N)
    
    # Vorticity
    zeta = (dV2_dxi1 - dV1_dxi2) / (sqrtG * R)
    f = 2.0 * omega * jnp.sin(lat)
    eta = zeta + f
    
    return eta, zeta, f


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def test_geostrophic_balance(N: int = 30, verbose: bool = True, sharding=None):
    """
    Test geostrophic balance using slope_reconstruction module.
    
    Args:
        N: Grid resolution
        verbose: Print detailed output
        sharding: Optional JAX sharding for multi-device execution
    
    Returns dict with test results.
    """
    t_wall_start = time.perf_counter()
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"GEOSTROPHIC BALANCE TEST (N={N})")
        print("=" * 70)
    
    # Setup
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
    
    # Convert to Cartesian
    Vx_all = jnp.zeros((6, N, N))
    Vy_all = jnp.zeros((6, N, N))
    Vz_all = jnp.zeros((6, N, N))
    X_all = jnp.zeros((6, N, N))
    Y_all = jnp.zeros((6, N, N))
    Z_all = jnp.zeros((6, N, N))
    
    for face in range(6):
        Vx, Vy, Vz = spherical_to_cartesian_velocity(
            u_lon[face], u_lat[face], lon[face], lat[face]
        )
        Vx_all = Vx_all.at[face].set(Vx)
        Vy_all = Vy_all.at[face].set(Vy)
        Vz_all = Vz_all.at[face].set(Vz)
        
        X, Y, Z = geom.xi_to_xyz(XI1, XI2, face)
        X_all = X_all.at[face].set(X * R)
        Y_all = Y_all.at[face].set(Y * R)
        Z_all = Z_all.at[face].set(Z * R)
    
    # Bernoulli function
    B = g * h + 0.5 * (Vx_all**2 + Vy_all**2 + Vz_all**2)
    
    # Apply sharding to distribute across devices
    if sharding is not None:
        B = jax.device_put(B, sharding)
        lat = jax.device_put(lat, sharding)
        Vx_all = jax.device_put(Vx_all, sharding)
        Vy_all = jax.device_put(Vy_all, sharding)
        Vz_all = jax.device_put(Vz_all, sharding)
        X_all = jax.device_put(X_all, sharding)
        Y_all = jax.device_put(Y_all, sharding)
        Z_all = jax.device_put(Z_all, sharding)
    
    t_setup = time.perf_counter() - t_setup_start
    
    # ========================================================================
    # GRADIENT COMPUTATION (timed)
    # ========================================================================
    t_grad_start = time.perf_counter()
    gradient_fn = make_gradient_fn(N, float(dx), XI1, XI2, R)
    dB_dX, dB_dY, dB_dZ = gradient_fn(B)
    jax.block_until_ready(dB_dX)
    t_grad = time.perf_counter() - t_grad_start
    
    # ========================================================================
    # ANALYTICAL VORTICITY TERM
    # ========================================================================
    eta_all = 2.0 * (omega + u0/R) * jnp.sin(lat)
    
    Cx_a = jnp.zeros((6, N, N))
    Cy_a = jnp.zeros((6, N, N))
    Cz_a = jnp.zeros((6, N, N))
    
    for face in range(6):
        Cx, Cy, Cz = compute_vorticity_term_cartesian(
            eta_all[face], Vx_all[face], Vy_all[face], Vz_all[face],
            X_all[face], Y_all[face], Z_all[face], R
        )
        Cx_a = Cx_a.at[face].set(Cx)
        Cy_a = Cy_a.at[face].set(Cy)
        Cz_a = Cz_a.at[face].set(Cz)
    
    # Reference magnitude for normalization
    vort_mag = jnp.sqrt(Cx_a**2 + Cy_a**2 + Cz_a**2)
    
    # ========================================================================
    # TEST 1: |∇B| vs |(ζ+f)k×V|_analytical
    # ========================================================================
    grad_B_mag = jnp.sqrt(dB_dX**2 + dB_dY**2 + dB_dZ**2)
    
    # Compute error field (exclude poles and equator)
    error_1 = []
    for face in range(6):
        lat_f = lat[face]
        mask = (jnp.abs(lat_f) > jnp.radians(10)) & (jnp.abs(lat_f) < jnp.radians(85))
        mask = mask & (vort_mag[face] > 1e-10)
        
        rel_error = jnp.abs(grad_B_mag[face] - vort_mag[face]) / vort_mag[face]
        valid = rel_error[mask]
        error_1.extend(valid.tolist())
    
    error_1 = np.array(error_1)
    L1_1 = np.mean(np.abs(error_1))
    L2_1 = np.sqrt(np.mean(error_1**2))
    Linf_1 = np.max(np.abs(error_1))
    
    # ========================================================================
    # TEST 2: |∇B + (ζ+f)k×V| residual
    # ========================================================================
    residual_mag = jnp.sqrt((dB_dX + Cx_a)**2 + (dB_dY + Cy_a)**2 + (dB_dZ + Cz_a)**2)
    
    error_2 = []
    for face in range(6):
        lat_f = lat[face]
        mask = (jnp.abs(lat_f) > jnp.radians(10)) & (jnp.abs(lat_f) < jnp.radians(85))
        mask = mask & (vort_mag[face] > 1e-10)
        
        rel_res = residual_mag[face] / vort_mag[face]
        valid = rel_res[mask]
        error_2.extend(valid.tolist())
    
    error_2 = np.array(error_2)
    L1_2 = np.mean(np.abs(error_2))
    L2_2 = np.sqrt(np.mean(error_2**2))
    Linf_2 = np.max(np.abs(error_2))
    
    # ========================================================================
    # TEST 3: η_numerical vs η_analytical (timed)
    # ========================================================================
    t_vort_start = time.perf_counter()
    
    sqrtG = jnp.array(geom.sqrtG)
    eta_analytical = 2.0 * (omega + u0/R) * jnp.sin(lat)
    
    error_3 = []
    for face in range(6):
        eta_num, _, _ = compute_absolute_vorticity_numerical(
            Vx_all[face], Vy_all[face], Vz_all[face],
            XI1, XI2, face, sqrtG[face],
            lat[face], omega, float(dx), N, R
        )
        
        lat_f = lat[face]
        mask = (jnp.abs(lat_f) > jnp.radians(10)) & (jnp.abs(lat_f) < jnp.radians(85))
        analytical = jnp.abs(eta_analytical[face])
        mask = mask & (analytical > 1e-10)
        
        rel_error = jnp.abs(eta_num - eta_analytical[face]) / analytical
        valid = rel_error[mask]
        error_3.extend(valid.tolist())
    
    # Block until vorticity computation complete
    jax.block_until_ready(eta_num)
    t_vort = time.perf_counter() - t_vort_start
    
    error_3 = np.array(error_3)
    L1_3 = np.mean(np.abs(error_3))
    L2_3 = np.sqrt(np.mean(error_3**2))
    Linf_3 = np.max(np.abs(error_3))
    
    # ========================================================================
    # TIMING SUMMARY
    # ========================================================================
    t_wall = time.perf_counter() - t_wall_start
    
    if verbose:
        print(f"\n  TIMING (wall clock):")
        print(f"    Setup:      {t_setup*1000:8.1f} ms")
        print(f"    Gradient:   {t_grad*1000:8.1f} ms")
        print(f"    Vorticity:  {t_vort*1000:8.1f} ms")
        print(f"    Total:      {t_wall*1000:8.1f} ms")
        
        print(f"\n  ERROR NORMS (relative):")
        print(f"    {'Test':<25} {'L1':>12} {'L2':>12} {'Linf':>12}")
        print(f"    {'-'*61}")
        print(f"    {'1: |∇B| vs analytical':<25} {L1_1:>12.2e} {L2_1:>12.2e} {Linf_1:>12.2e}")
        print(f"    {'2: momentum residual':<25} {L1_2:>12.2e} {L2_2:>12.2e} {Linf_2:>12.2e}")
        print(f"    {'3: vorticity':<25} {L1_3:>12.2e} {L2_3:>12.2e} {Linf_3:>12.2e}")
        print(f"\n    Expected truncation ~ dx² = {dx**2:.2e}")
    
    return {
        'N': N,
        'dx': float(dx),
        'L1_test1': L1_1, 'L2_test1': L2_1, 'Linf_test1': Linf_1,
        'L1_test2': L1_2, 'L2_test2': L2_2, 'Linf_test2': Linf_2,
        'L1_test3': L1_3, 'L2_test3': L2_3, 'Linf_test3': Linf_3,
        'expected_truncation': dx**2,
        't_setup_ms': t_setup * 1000,
        't_grad_ms': t_grad * 1000,
        't_vort_ms': t_vort * 1000,
        't_wall_ms': t_wall * 1000,
    }


# =============================================================================
# CONVERGENCE STUDY
# =============================================================================

def run_convergence_study(resolutions: list, sharding=None):
    """Run convergence study across resolutions."""
    print(f"\n{'='*70}")
    print("CONVERGENCE STUDY")
    print(f"  Precision: {args.precision}, Device: {args.device}, Num devices: {args.num_devices}")
    print(f"{'='*70}")
    
    results = []
    for N in resolutions:
        r = test_geostrophic_balance(N, sharding=sharding)
        results.append(r)
    
    # Summary table
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    
    # L2 norm summary (most common for convergence)
    print(f"\n  L2 Error Norms:")
    print(f"    {'N':>6} {'dx':>10} {'Test1':>12} {'Test2':>12} {'Test3':>12} {'dx²':>12}")
    print(f"    {'-'*66}")
    
    for r in results:
        print(f"    {r['N']:>6} {r['dx']:>10.4f} {r['L2_test1']:>12.2e} {r['L2_test2']:>12.2e} {r['L2_test3']:>12.2e} {r['expected_truncation']:>12.2e}")
    
    # Linf norm summary
    print(f"\n  Linf Error Norms:")
    print(f"    {'N':>6} {'dx':>10} {'Test1':>12} {'Test2':>12} {'Test3':>12}")
    print(f"    {'-'*54}")
    
    for r in results:
        print(f"    {r['N']:>6} {r['dx']:>10.4f} {r['Linf_test1']:>12.2e} {r['Linf_test2']:>12.2e} {r['Linf_test3']:>12.2e}")
    
    # Timing summary
    print(f"\n  Timing (wall clock, ms):")
    print(f"    {'N':>6} {'Setup':>10} {'Gradient':>10} {'Vorticity':>10} {'Total':>10}")
    print(f"    {'-'*50}")
    
    for r in results:
        print(f"    {r['N']:>6} {r['t_setup_ms']:>10.0f} {r['t_grad_ms']:>10.0f} {r['t_vort_ms']:>10.0f} {r['t_wall_ms']:>10.0f}")
    
    # Convergence rates (using L2 norm)
    if len(results) >= 2:
        r1, r2 = results[-2], results[-1]
        
        rate1 = np.log(r1['L2_test1'] / r2['L2_test1']) / np.log(r1['dx'] / r2['dx'])
        rate2 = np.log(r1['L2_test2'] / r2['L2_test2']) / np.log(r1['dx'] / r2['dx'])
        rate3 = np.log(r1['L2_test3'] / r2['L2_test3']) / np.log(r1['dx'] / r2['dx'])
        
        print(f"\n  Convergence rates (L2 norm, N={r1['N']}→{r2['N']}):")
        print(f"    Test 1: {rate1:.2f}  Test 2: {rate2:.2f}  Test 3: {rate3:.2f}")
        print(f"    Expected: 2.0")
        
        all_pass = rate1 > 1.8 and rate2 > 1.8 and rate3 > 1.8
        print(f"\n  Result: {'PASS' if all_pass else 'FAIL'}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if args.N is not None:
        test_geostrophic_balance(args.N, sharding=sharding)
    else:
        resolutions = [int(n) for n in args.resolutions.split(',')]
        run_convergence_study(resolutions, sharding=sharding)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}\n")
