"""
Comprehensive diagnostic plots for SWE solver validation.

Creates:
1. Vorticity term error (numerical - analytical) at t=0 and after timesteps
2. Geostrophic imbalance (|∇B + η×V|) at t=0
3. Shows which gradient method is being used
"""

import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Solvers.fv_plr_cubesphere_swe import (
    CubedSphereSWE, 
    compute_pressure_gradient_cartesian,
    compute_coriolis_cartesian
)
from Solvers.halo_exchange import exchange_scalar_halos_v2


def plot_6faces(data, title, filename, cmap='RdBu_r', symmetric=True, vmin=None, vmax=None):
    """Plot 6 cubed-sphere faces as flat images."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    if symmetric and vmin is None:
        absmax = np.max(np.abs(data))
        vmin, vmax = -absmax, absmax
    elif vmin is None:
        vmin, vmax = np.min(data), np.max(data)
    
    face_names = ['Face 0 (N)', 'Face 1 (E)', 'Face 2', 'Face 3', 'Face 4 (W)', 'Face 5 (S)']
    
    for face in range(6):
        ax = axes[face // 3, face % 3]
        im = ax.imshow(data[face], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{face_names[face]}')
        ax.set_xlabel('j')
        ax.set_ylabel('i')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def compute_vorticity_error(solver, state, analytical=True):
    """
    Compute vorticity term error: numerical - analytical.
    
    For solid body rotation:
        η = (2Ω + 2u0/R) * sin(lat) = (2Ω + 2u0/R) * Z
        -η(r̂ × V) is the vorticity acceleration
    """
    R = solver.R
    omega = solver.omega
    u0 = solver.u0
    
    # Compute numerical vorticity term
    numerical_mag = np.zeros((6, solver.N, solver.N))
    analytical_mag = np.zeros((6, solver.N, solver.N))
    error = np.zeros((6, solver.N, solver.N))
    
    for face in range(6):
        # Numerical
        cor_x, cor_y, cor_z = compute_coriolis_cartesian(
            state.Vx[face], state.Vy[face], state.Vz[face],
            solver.X_all[face], solver.Y_all[face], solver.Z_all[face],
            omega, u0=u0, R=R
        )
        num_mag = np.sqrt(np.array(cor_x)**2 + np.array(cor_y)**2 + np.array(cor_z)**2)
        
        # Analytical: |η × V| = η * |V| (since V ⊥ r̂)
        # η = 2(Ω + u0/R) * sin(lat) = 2(Ω + u0/R) * Z
        X, Y, Z = np.array(solver.X_all[face]), np.array(solver.Y_all[face]), np.array(solver.Z_all[face])
        eta = 2.0 * (omega + u0/R) * Z
        
        Vx, Vy, Vz = np.array(state.Vx[face]), np.array(state.Vy[face]), np.array(state.Vz[face])
        V_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        
        # |r̂ × V| for V tangent to sphere = |V| * sin(angle) ≈ |V| (since V ⊥ r̂)
        # Actually compute it exactly:
        cross_x = Y * Vz - Z * Vy
        cross_y = Z * Vx - X * Vz
        cross_z = X * Vy - Y * Vx
        cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        
        ana_mag = np.abs(eta) * cross_mag
        
        numerical_mag[face] = num_mag
        analytical_mag[face] = ana_mag
        error[face] = num_mag - ana_mag
    
    return numerical_mag, analytical_mag, error


def compute_geostrophic_imbalance(solver, state):
    """
    Compute geostrophic imbalance: ∇B + η×V (should be zero for steady state).
    
    Returns magnitude of the imbalance.
    """
    R = solver.R
    g = solver.g
    omega = solver.omega
    u0 = solver.u0
    dx = solver.dx
    N = solver.N
    
    # Extend geometry
    XI1_ghosts = jnp.pad(solver.XI1_all, ((0,0), (1,1), (1,1)), mode='edge')
    XI2_ghosts = jnp.pad(solver.XI2_all, ((0,0), (1,1), (1,1)), mode='edge')
    
    # Halo exchange
    h_ghosts = exchange_scalar_halos_v2(state.h, N, solver.halo_exchange)
    Vx_ghosts = exchange_scalar_halos_v2(state.Vx, N, solver.halo_exchange)
    Vy_ghosts = exchange_scalar_halos_v2(state.Vy, N, solver.halo_exchange)
    Vz_ghosts = exchange_scalar_halos_v2(state.Vz, N, solver.halo_exchange)
    
    imbalance_mag = np.zeros((6, N, N))
    grad_B_mag = np.zeros((6, N, N))
    
    for face in range(6):
        # Bernoulli function B = gh + K
        V_sq = Vx_ghosts[face]**2 + Vy_ghosts[face]**2 + Vz_ghosts[face]**2
        K_ghosts = 0.5 * V_sq
        B_ghosts = g * h_ghosts[face] + K_ghosts
        
        # Bernoulli gradient
        dB_dX, dB_dY, dB_dZ = compute_pressure_gradient_cartesian(
            B_ghosts / g, XI1_ghosts[face], XI2_ghosts[face],
            face, dx, N, g, R
        )
        
        # Vorticity term
        cor_x, cor_y, cor_z = compute_coriolis_cartesian(
            state.Vx[face], state.Vy[face], state.Vz[face],
            solver.X_all[face], solver.Y_all[face], solver.Z_all[face],
            omega, u0=u0, R=R
        )
        
        # Imbalance: ∇B + (-η × V) = ∇B - η × V
        # For balance: ∇B = η × V, so ∇B + coriolis = 0 if coriolis = -η × V
        total_x = np.array(dB_dX) + np.array(cor_x)
        total_y = np.array(dB_dY) + np.array(cor_y)
        total_z = np.array(dB_dZ) + np.array(cor_z)
        
        imbalance_mag[face] = np.sqrt(total_x**2 + total_y**2 + total_z**2)
        grad_B_mag[face] = np.sqrt(np.array(dB_dX)**2 + np.array(dB_dY)**2 + np.array(dB_dZ)**2)
    
    return imbalance_mag, grad_B_mag


def main():
    print("=" * 70)
    print("DIAGNOSTIC PLOTS FOR SWE SOLVER")
    print("=" * 70)
    
    N = 30
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['solver']['N'] = N
    
    # Create solver
    solver = CubedSphereSWE(N, config)
    
    # =========================================================================
    # PART 1: Check what gradient method is being used
    # =========================================================================
    print("\n" + "=" * 70)
    print("GRADIENT METHOD CHECK")
    print("=" * 70)
    
    # Read the source file to check
    swe_file = Path(__file__).parent.parent.parent / "Solvers" / "fv_plr_cubesphere_swe.py"
    with open(swe_file) as f:
        content = f.read()
    
    if "compute_limited_slopes" in content and "def compute_pressure_gradient_cartesian" in content:
        # Find the function and check what it uses
        import re
        func_match = re.search(r'def compute_pressure_gradient_cartesian.*?(?=\ndef |\Z)', content, re.DOTALL)
        if func_match:
            func_code = func_match.group()
            if "compute_limited_slopes" in func_code:
                print("⚠ Using FV-PLR (limited slopes) for pressure gradient")
                gradient_method = "FV-PLR (MC limiter)"
            else:
                print("✓ Using unlimited central differences for pressure gradient")
                gradient_method = "Central differences (unlimited)"
                
            # Check for the specific comment
            if "UNLIMITED central difference" in func_code:
                print("  (Confirmed: unlimited central differences)")
    
    # =========================================================================
    # PART 2: Vorticity term at t=0
    # =========================================================================
    print("\n" + "=" * 70)
    print("VORTICITY TERM ERROR AT t=0")
    print("=" * 70)
    
    state = solver.initialize(config, test_case='testcase2')
    
    num_vort, ana_vort, vort_error = compute_vorticity_error(solver, state)
    
    print(f"\n  Numerical vorticity magnitude: [{num_vort.min():.4e}, {num_vort.max():.4e}]")
    print(f"  Analytical vorticity magnitude: [{ana_vort.min():.4e}, {ana_vort.max():.4e}]")
    print(f"  Error (numerical - analytical): [{vort_error.min():.4e}, {vort_error.max():.4e}]")
    print(f"  Max relative error: {np.abs(vort_error).max() / ana_vort.max():.2e}")
    
    plot_6faces(vort_error, 
                f'Vorticity Term Error at t=0 (numerical - analytical) [m/s²]\nMax error: {np.abs(vort_error).max():.2e}',
                'vorticity_error_t0.png')
    
    # =========================================================================
    # PART 3: Vorticity after timesteps
    # =========================================================================
    print("\n" + "=" * 70)
    print("VORTICITY TERM ERROR AFTER 10 TIMESTEPS")
    print("=" * 70)
    
    state = solver.initialize(config, test_case='testcase2')
    dt = 100.0
    for step in range(10):
        state = solver.step(state, dt)
    
    num_vort, ana_vort, vort_error = compute_vorticity_error(solver, state)
    
    print(f"\n  After 10 steps (t = {10*dt}s):")
    print(f"  Numerical vorticity magnitude: [{num_vort.min():.4e}, {num_vort.max():.4e}]")
    print(f"  Analytical vorticity magnitude: [{ana_vort.min():.4e}, {ana_vort.max():.4e}]")
    print(f"  Error (numerical - analytical): [{vort_error.min():.4e}, {vort_error.max():.4e}]")
    print(f"  Max relative error: {np.abs(vort_error).max() / ana_vort.max():.2e}")
    
    plot_6faces(vort_error,
                f'Vorticity Term Error after 10 steps (t={10*dt}s) [m/s²]\nMax error: {np.abs(vort_error).max():.2e}',
                'vorticity_error_t10.png')
    
    # =========================================================================
    # PART 4: Geostrophic imbalance at t=0
    # =========================================================================
    print("\n" + "=" * 70)
    print("GEOSTROPHIC IMBALANCE AT t=0")
    print("=" * 70)
    
    state = solver.initialize(config, test_case='testcase2')
    
    imbalance, grad_B = compute_geostrophic_imbalance(solver, state)
    relative_imbalance = imbalance / np.maximum(grad_B, 1e-10)
    
    print(f"\n  Bernoulli gradient |∇B|: [{grad_B.min():.4e}, {grad_B.max():.4e}] m/s²")
    print(f"  Imbalance |∇B + η×V|: [{imbalance.min():.4e}, {imbalance.max():.4e}] m/s²")
    print(f"  Relative imbalance: [{relative_imbalance.min():.2%}, {relative_imbalance.max():.2%}]")
    print(f"\n  NOTE: This is the GRADIENT imbalance, not total energy")
    print(f"  Using: {gradient_method}")
    
    plot_6faces(imbalance,
                f'Geostrophic Imbalance |∇B + η×V| at t=0 [m/s²]\nMax: {imbalance.max():.2e}, Method: {gradient_method}',
                'geostrophic_imbalance_t0.png',
                cmap='hot', symmetric=False, vmin=0)
    
    plot_6faces(100 * relative_imbalance,
                f'Relative Geostrophic Imbalance at t=0 [%]\nMax: {100*relative_imbalance.max():.1f}%',
                'geostrophic_imbalance_relative_t0.png',
                cmap='hot', symmetric=False, vmin=0, vmax=100)
    
    # =========================================================================
    # PART 5: Show where max imbalance occurs
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOCATION OF MAXIMUM IMBALANCE")
    print("=" * 70)
    
    max_idx = np.unravel_index(np.argmax(imbalance), imbalance.shape)
    face, i, j = max_idx
    
    X = float(solver.X_all[face, i, j])
    Y = float(solver.Y_all[face, i, j])
    Z = float(solver.Z_all[face, i, j])
    lat = np.degrees(np.arcsin(Z))
    lon = np.degrees(np.arctan2(Y, X))
    
    print(f"\n  Maximum imbalance at:")
    print(f"    Face {face}, (i,j) = ({i},{j})")
    print(f"    (lat, lon) = ({lat:.1f}°, {lon:.1f}°)")
    print(f"    (X, Y, Z) = ({X:.4f}, {Y:.4f}, {Z:.4f})")
    
    # Check if near corner
    corner_dist = min(i, j, N-1-i, N-1-j)
    print(f"    Distance from corner: {corner_dist} cells")
    if corner_dist <= 3:
        print(f"    ⚠ Near cubed-sphere corner (Putman & Lin 2007 singularity)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Gradient method: {gradient_method}
  
  Vorticity term (at t=0):
    Max error: {np.abs(vort_error).max():.2e} m/s²
    Relative: {np.abs(vort_error).max() / ana_vort.max():.2e}
  
  Geostrophic imbalance (at t=0):
    Max absolute: {imbalance.max():.2e} m/s²
    Max relative: {100*relative_imbalance.max():.1f}%
    Location: Face {face}, near {'corner' if corner_dist <= 3 else 'interior'}
    
  Reference: Putman & Lin (2007) note that cubed-sphere corners have
  120° coordinate intersections, causing geometric singularities.
""")


if __name__ == "__main__":
    main()

