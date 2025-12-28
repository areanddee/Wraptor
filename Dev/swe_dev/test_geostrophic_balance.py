"""
Test geostrophic balance: g*∇h should equal f*V for Test Case 2.

For steady geostrophic flow:
    g * ∇h + f × V = 0
    
So: g * ∇h = -f × V (pressure gradient balances Coriolis)
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

from Solvers.fv_plr_cubesphere_swe import (
    CubedSphereSWE, 
    compute_pressure_gradient_cartesian,
    compute_coriolis_cartesian
)
from Solvers.halo_exchange import exchange_scalar_halos_v2


def main():
    print("=" * 70)
    print("GEOSTROPHIC BALANCE TEST")
    print("=" * 70)
    
    N = 30
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['solver']['N'] = N
    
    # Create solver
    solver = CubedSphereSWE(N, config)
    
    # Initialize with Test Case 2 (steady geostrophic)
    state = solver.initialize(config, test_case='testcase2')
    
    R = solver.R
    g = solver.g
    omega = solver.omega
    dx = solver.dx
    
    print(f"\nParameters:")
    print(f"  R = {R:.3e} m")
    print(f"  g = {g:.2f} m/s²")
    print(f"  omega = {omega:.6e} rad/s")
    
    # Extend geometry
    XI1_ghosts = jnp.pad(solver.XI1_all, ((0,0), (1,1), (1,1)), mode='edge')
    XI2_ghosts = jnp.pad(solver.XI2_all, ((0,0), (1,1), (1,1)), mode='edge')
    
    # Halo exchange h
    h_ghosts = exchange_scalar_halos_v2(state.h, N, solver.halo_exchange)
    
    # Compute for each face
    print("\n" + "=" * 70)
    print("GEOSTROPHIC BALANCE PER FACE")
    print("=" * 70)
    
    # Extend V for K computation
    Vx_ghosts = exchange_scalar_halos_v2(state.Vx, N, solver.halo_exchange)
    Vy_ghosts = exchange_scalar_halos_v2(state.Vy, N, solver.halo_exchange)
    Vz_ghosts = exchange_scalar_halos_v2(state.Vz, N, solver.halo_exchange)
    
    for face in range(6):
        # Compute Bernoulli function B = g*h + K = g*h + |V|²/2
        V_sq = Vx_ghosts[face]**2 + Vy_ghosts[face]**2 + Vz_ghosts[face]**2
        K_ghosts = 0.5 * V_sq
        B_ghosts = g * h_ghosts[face] + K_ghosts
        
        # Bernoulli gradient: ∇B = g∇h + ∇K
        # Note: compute_pressure_gradient_cartesian computes -g∇(input)
        # So we pass B/g to get -∇B
        dB_dX, dB_dY, dB_dZ = compute_pressure_gradient_cartesian(
            B_ghosts / g, XI1_ghosts[face], XI2_ghosts[face],
            face, dx, N, g, R
        )
        
        # Vorticity term: -η (r̂ × V) with full absolute vorticity
        cor_x, cor_y, cor_z = compute_coriolis_cartesian(
            state.Vx[face], state.Vy[face], state.Vz[face],
            solver.X_all[face], solver.Y_all[face], solver.Z_all[face],
            omega, u0=solver.u0, R=R
        )
        
        # Total momentum tendency (should be near zero for steady state)
        # Vector-invariant form: dV/dt = -∇B + ζ_a × V
        # where ζ_a × V ≈ f × V for small relative vorticity
        total_x = dB_dX + cor_x
        total_y = dB_dY + cor_y
        total_z = dB_dZ + cor_z
        
        # Magnitudes
        B_grad_mag = jnp.sqrt(dB_dX**2 + dB_dY**2 + dB_dZ**2)
        cor_mag = jnp.sqrt(cor_x**2 + cor_y**2 + cor_z**2)
        total_mag = jnp.sqrt(total_x**2 + total_y**2 + total_z**2)
        
        print(f"\nFace {face}:")
        print(f"  Bernoulli gradient: [{float(B_grad_mag.min()):.4e}, {float(B_grad_mag.max()):.4e}] m/s²")
        print(f"  Coriolis:           [{float(cor_mag.min()):.4e}, {float(cor_mag.max()):.4e}] m/s²")
        print(f"  Imbalance:          [{float(total_mag.min()):.4e}, {float(total_mag.max()):.4e}] m/s²")
        print(f"  Relative imbalance: {float(total_mag.max() / B_grad_mag.max()):.2%}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL IMBALANCE")
    print("=" * 70)
    
    # Compute total imbalance
    all_imbalance = []
    all_B_grad = []
    
    for face in range(6):
        V_sq = Vx_ghosts[face]**2 + Vy_ghosts[face]**2 + Vz_ghosts[face]**2
        K_ghosts = 0.5 * V_sq
        B_ghosts = g * h_ghosts[face] + K_ghosts
        
        dB_dX, dB_dY, dB_dZ = compute_pressure_gradient_cartesian(
            B_ghosts / g, XI1_ghosts[face], XI2_ghosts[face],
            face, dx, N, g, R
        )
        cor_x, cor_y, cor_z = compute_coriolis_cartesian(
            state.Vx[face], state.Vy[face], state.Vz[face],
            solver.X_all[face], solver.Y_all[face], solver.Z_all[face],
            omega, u0=solver.u0, R=R
        )
        
        total_x = dB_dX + cor_x
        total_y = dB_dY + cor_y
        total_z = dB_dZ + cor_z
        
        total_mag = jnp.sqrt(total_x**2 + total_y**2 + total_z**2)
        B_grad_mag = jnp.sqrt(dB_dX**2 + dB_dY**2 + dB_dZ**2)
        
        all_imbalance.append(total_mag)
        all_B_grad.append(B_grad_mag)
    
    all_imbalance = jnp.stack(all_imbalance)
    all_B_grad = jnp.stack(all_B_grad)
    
    print(f"\n  Max imbalance: {float(all_imbalance.max()):.4e} m/s²")
    print(f"  Max Bernoulli gradient: {float(all_B_grad.max()):.4e} m/s²")
    print(f"  Relative imbalance: {float(all_imbalance.max() / all_B_grad.max()):.2%}")
    
    if float(all_imbalance.max() / all_B_grad.max()) < 0.01:
        print(f"\n✓ GOOD: Geostrophic balance within 1%")
    else:
        print(f"\n✗ BAD: Significant geostrophic imbalance")


if __name__ == "__main__":
    main()

