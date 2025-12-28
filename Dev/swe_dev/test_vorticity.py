"""
Test absolute vorticity computation for solid body rotation.

Analytical solution (Williamson et al. 1992, eq. 94):
    ζ_a = (2*u0/R + 2*Omega) * cos(lat)
    
where:
- u0 = max zonal velocity at equator [m/s]
- R = planet radius [m]
- Omega = planet rotation rate [rad/s]
- lat = latitude [rad]
"""

import sys
from pathlib import Path
import yaml

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Solvers.fv_plr_cubesphere_swe import CubedSphereSWE
from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams


def compute_relative_vorticity_cartesian(Vx, Vy, Vz, X, Y, Z, geometry, planet):
    """
    Compute relative vorticity ζ = (∇ × V) · r_hat from Cartesian velocities.
    
    For a velocity field on the sphere, the relative vorticity is the 
    radial component of the curl.
    
    Returns:
        zeta: (6, N, N) relative vorticity [1/s]
    """
    # This is a simplified computation using finite differences
    # For solid body rotation V = u0 * (-Y, X, 0), the vorticity should be constant
    
    # Actually, let's compute it analytically for the test
    # For V = u0 * (-Y, X, 0) on unit sphere:
    # ∇ × V = (∂Vz/∂y - ∂Vy/∂z, ∂Vx/∂z - ∂Vz/∂x, ∂Vy/∂x - ∂Vx/∂y)
    # For V = u0 * (-Y, X, 0):
    # ∂Vy/∂x = u0, ∂Vx/∂y = -u0
    # So (∇ × V)_z = u0 - (-u0) = 2*u0
    # But we need the radial component...
    
    # For solid body rotation with angular velocity ω around Z-axis:
    # V = ω × r = ω * (-Y, X, 0)
    # The relative vorticity is ζ = 2ω
    # At equator: u = ω*R, so ω = u0/R, and ζ = 2*u0/R
    # At latitude θ: u = u0*cos(θ), and ζ = 2*(u0/R)*cos(θ)... wait that's not right
    
    # Actually for solid body rotation:
    # ζ = (1/(R*cos(lat))) * ∂(u*cos(lat))/∂lat - (1/(R*cos(lat))) * ∂v/∂lon
    # For u = u0*cos(lat), v = 0:
    # ∂(u*cos(lat))/∂lat = ∂(u0*cos²(lat))/∂lat = -2*u0*cos(lat)*sin(lat)
    # ζ = -2*u0*sin(lat)/(R)
    # Hmm, that's different...
    
    # Let me just return the numerical curl
    return None  # Placeholder


def main():
    print("=" * 70)
    print("ABSOLUTE VORTICITY TEST")
    print("=" * 70)
    
    N = 30
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['solver']['N'] = N
    
    # Create solver
    solver = CubedSphereSWE(N, config)
    
    # Initialize with Test Case 1 (or 2 for geostrophic)
    # Both use u0 = 40 m/s or from rotation period
    rotation_period_days = 12.0
    u0 = 2.0 * np.pi * solver.planet.R_sphere / (rotation_period_days * 86400.0)
    R = solver.planet.R_sphere
    Omega = solver.planet.omega
    
    print(f"\nParameters:")
    print(f"  R = {R:.3e} m")
    print(f"  Omega = {Omega:.3e} rad/s")
    print(f"  u0 = {u0:.2f} m/s (from 12-day rotation)")
    
    # Analytical absolute vorticity: (2*u0/R + 2*Omega) * cos(lat)
    # Note: 2*u0/R is the relative vorticity contribution
    #       2*Omega is the planetary vorticity (f = 2*Omega*sin(lat) for f, but for curl of planet it's 2*Omega)
    
    # Actually, let me re-read Williamson eq 94 more carefully
    # The absolute vorticity for solid body rotation is:
    # η = ζ + f = (relative vorticity) + (Coriolis parameter)
    # For solid body rotation at angle α=0 (around pole):
    #   ζ = 2*u0/R  (uniform for solid body rotation!)
    #   f = 2*Omega*sin(lat)
    # So η = 2*u0/R + 2*Omega*sin(lat)
    
    # But you said: (2*u0/R + 2*Omega)*cos(theta)
    # This might be a different formula... let me compute both and compare
    
    # Get lat/lon for all faces
    lat_all, lon_all = solver.geometry.get_lat_lon_all_faces()
    lat_all = jnp.array(lat_all)
    
    # Formula 1: η = 2*u0/R + 2*Omega*sin(lat)  [standard]
    zeta_rel = 2.0 * u0 / R
    f = 2.0 * Omega * jnp.sin(lat_all)
    eta_standard = zeta_rel + f
    
    # Formula 2: η = (2*u0/R + 2*Omega) * cos(lat)  [user's formula]
    eta_user = (2.0 * u0 / R + 2.0 * Omega) * jnp.cos(lat_all)
    
    print(f"\n--- Relative Vorticity (should be uniform for solid body rotation) ---")
    print(f"  ζ = 2*u0/R = {zeta_rel:.6e} 1/s")
    
    print(f"\n--- Coriolis parameter f = 2*Omega*sin(lat) ---")
    print(f"  f at equator (lat=0): {float(2*Omega*np.sin(0)):.6e}")
    print(f"  f at 45°N: {float(2*Omega*np.sin(np.pi/4)):.6e}")
    print(f"  f at pole (lat=90°): {float(2*Omega*np.sin(np.pi/2)):.6e}")
    
    print(f"\n--- Absolute Vorticity (standard formula: ζ + f) ---")
    print(f"  η at equator: {float(zeta_rel + 2*Omega*np.sin(0)):.6e}")
    print(f"  η at 45°N: {float(zeta_rel + 2*Omega*np.sin(np.pi/4)):.6e}")
    print(f"  η at pole: {float(zeta_rel + 2*Omega*np.sin(np.pi/2)):.6e}")
    
    print(f"\n--- Your formula: (2*u0/R + 2*Omega) * cos(lat) ---")
    print(f"  η at equator: {float((2*u0/R + 2*Omega)*np.cos(0)):.6e}")
    print(f"  η at 45°N: {float((2*u0/R + 2*Omega)*np.cos(np.pi/4)):.6e}")
    print(f"  η at pole: {float((2*u0/R + 2*Omega)*np.cos(np.pi/2)):.6e}")
    
    # Now let's compute the Coriolis term from the solver and compare
    print("\n" + "=" * 70)
    print("CORIOLIS TERM FROM SOLVER")
    print("=" * 70)
    
    # Initialize state
    state = solver.initialize(config, test_case='testcase1')
    
    # Get Coriolis acceleration for one face
    from Solvers.fv_plr_cubesphere_swe import compute_coriolis_cartesian
    
    face = 0
    cor_x, cor_y, cor_z = compute_coriolis_cartesian(
        state.Vx[face], state.Vy[face], state.Vz[face],
        solver.X_all[face], solver.Y_all[face], solver.Z_all[face],
        Omega
    )
    
    # Coriolis magnitude
    cor_mag = jnp.sqrt(cor_x**2 + cor_y**2 + cor_z**2)
    
    print(f"\nCoriolis acceleration magnitude (face {face}):")
    print(f"  min: {float(cor_mag.min()):.6e} m/s²")
    print(f"  max: {float(cor_mag.max()):.6e} m/s²")
    
    # Expected: f * |V| where f = 2*Omega*sin(lat)
    V_mag = jnp.sqrt(state.Vx[face]**2 + state.Vy[face]**2 + state.Vz[face]**2)
    lat_face = lat_all[face]
    f_face = 2.0 * Omega * jnp.sin(lat_face)
    expected_cor = jnp.abs(f_face) * V_mag
    
    print(f"\nExpected Coriolis magnitude (f * |V|):")
    print(f"  min: {float(expected_cor.min()):.6e} m/s²")
    print(f"  max: {float(expected_cor.max()):.6e} m/s²")
    
    diff = jnp.abs(cor_mag - expected_cor)
    print(f"\nDifference:")
    print(f"  max: {float(diff.max()):.6e}")
    print(f"  relative max: {float(diff.max() / expected_cor.max()):.6e}")


if __name__ == "__main__":
    main()

