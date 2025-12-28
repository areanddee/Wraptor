"""
Simple test of flux divergence on cubed-sphere.

The simplest possible test: compute ∇·(h*V) for a known h and V,
compare to analytical value.

Test case: h = 1 (constant), V = solid body rotation
Analytical: ∇·(h*V) = ∇·V = 0 (incompressible flow)

If this doesn't give zero, something is fundamentally wrong.
"""

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Solvers.geometry.cubesphere import CubedSphereGeometry
from Solvers.halo_exchange import (
    create_communication_schedule,
    make_halo_exchange,
    exchange_scalar_halos_v2,
)
# Use the WORKING functions from the advection solver!
from Solvers.fv_plr_cubesphere_adv import cartesian_to_contravariant, compute_jacobian_face


def compute_divergence_upwind(h_all, Vx_all, Vy_all, Vz_all, geometry, halo_fn):
    """
    Compute ∇·(h*V) using simple upwind flux.
    
    Flux: F = h * u (where u is contravariant velocity)
    Upwind: use value from upwind cell
    """
    N = geometry.N
    dx = geometry.dx
    sqrtG_all = jnp.array(geometry.sqrtG)
    XI1 = jnp.array(geometry.XI1)
    XI2 = jnp.array(geometry.XI2)
    
    # Halo exchange
    h_ghosts_all = exchange_scalar_halos_v2(h_all, N, halo_fn)
    Vx_ghosts_all = exchange_scalar_halos_v2(Vx_all, N, halo_fn)
    Vy_ghosts_all = exchange_scalar_halos_v2(Vy_all, N, halo_fn)
    Vz_ghosts_all = exchange_scalar_halos_v2(Vz_all, N, halo_fn)
    
    div_h = jnp.zeros_like(h_all)
    
    for face in range(6):
        h_ghosts = h_ghosts_all[face]
        Vx_ghosts = Vx_ghosts_all[face]
        Vy_ghosts = Vy_ghosts_all[face]
        Vz_ghosts = Vz_ghosts_all[face]
        R = 6.371e6  # Must match the R in compute_jacobian_face
        sqrtG = sqrtG_all[face] * R**2  # Scale to physical units
        
        # Compute contravariant velocities at cell centers
        XI1_ghosts = jnp.pad(XI1, ((1, 1), (1, 1)), mode='edge')
        XI2_ghosts = jnp.pad(XI2, ((1, 1), (1, 1)), mode='edge')
        
        u1_ghosts, u2_ghosts = cartesian_to_contravariant(
            Vx_ghosts, Vy_ghosts, Vz_ghosts,
            XI1_ghosts, XI2_ghosts, face
        )
        
        # Pad sqrtG (already scaled by R²)
        sqrtG_ghosts = jnp.pad(sqrtG, ((1, 1), (1, 1)), mode='edge')
        
        # Interior values
        h = h_ghosts[1:-1, 1:-1]
        u1 = u1_ghosts[1:-1, 1:-1]
        u2 = u2_ghosts[1:-1, 1:-1]
        
        # =====================================================================
        # ξ¹ direction: compute flux at ALL N+1 interfaces with SAME formula
        # Interface i+1/2 is between cell i and cell i+1
        # For cell indices 0..N-1, interfaces are at i = -1/2, 1/2, ..., N-1/2
        # In ghost array (size N+2), this means interface i+1/2 is between 
        # ghost index i+1 and i+2
        # =====================================================================
        
        # N+1 interfaces in ξ¹ direction, spanning the N interior cells
        # Interface at i+1/2: average of cells [i+1] and [i+2] in ghost array
        # For i = -1 to N-1: ghost indices are 0:N+1 and 1:N+2
        u1_face = 0.5 * (u1_ghosts[0:N+1, 1:N+1] + u1_ghosts[1:N+2, 1:N+1])
        sqrtG_face = 0.5 * (sqrtG_ghosts[0:N+1, 1:N+1] + sqrtG_ghosts[1:N+2, 1:N+1])
        
        # Upwind h value: use h from cell where flow comes from
        h_L = h_ghosts[0:N+1, 1:N+1]  # left of interface
        h_R = h_ghosts[1:N+2, 1:N+1]  # right of interface
        h_face = jnp.where(u1_face > 0, h_L, h_R)
        
        # Flux at all N+1 interfaces: shape (N+1, N)
        F1 = sqrtG_face * u1_face * h_face
        
        # =====================================================================
        # ξ² direction: compute flux at ALL N+1 interfaces with SAME formula
        # =====================================================================
        
        u2_face = 0.5 * (u2_ghosts[1:N+1, 0:N+1] + u2_ghosts[1:N+1, 1:N+2])
        sqrtG_face2 = 0.5 * (sqrtG_ghosts[1:N+1, 0:N+1] + sqrtG_ghosts[1:N+1, 1:N+2])
        
        h_L2 = h_ghosts[1:N+1, 0:N+1]
        h_R2 = h_ghosts[1:N+1, 1:N+2]
        h_face2 = jnp.where(u2_face > 0, h_L2, h_R2)
        
        # Flux at all N+1 interfaces: shape (N, N+1)
        F2 = sqrtG_face2 * u2_face * h_face2
        
        # =====================================================================
        # Divergence: (F[i+1/2] - F[i-1/2]) / (sqrtG * dx)
        # F1 has shape (N+1, N), F2 has shape (N, N+1)
        # =====================================================================
        
        dF1_dx1 = (F1[1:, :] - F1[:-1, :]) / dx  # shape (N, N)
        dF2_dx2 = (F2[:, 1:] - F2[:, :-1]) / dx  # shape (N, N)
        
        div_face = (dF1_dx1 + dF2_dx2) / sqrtG
        div_h = div_h.at[face].set(div_face)
    
    return div_h


def main():
    print("=" * 70)
    print("TEST: Divergence of Incompressible Flow")
    print("=" * 70)
    
    print("\nTest case: h = 1 (constant), V = solid body rotation")
    print("Analytical: ∇·(h*V) = ∇·V = 0 (incompressible)")
    print("\nThis tests if the flux divergence gives zero for divergence-free flow.")
    
    resolutions = [10, 20, 40, 80]
    
    for N in resolutions:
        print(f"\n--- N = {N} ---")
        
        geometry = CubedSphereGeometry.create(N)
        
        # h = 1 everywhere
        h_all = jnp.ones((6, N, N))
        
        # Solid body rotation: V = u0 * (ẑ × r̂) = u0 * (-Y, X, 0)
        u0 = 1.0
        X_all, Y_all, Z_all = geometry.get_xyz_all_faces()
        Vx_all = -u0 * jnp.array(Y_all)
        Vy_all = u0 * jnp.array(X_all)
        Vz_all = jnp.zeros((6, N, N))
        
        # Halo exchange
        schedule = create_communication_schedule()
        halo_fn = make_halo_exchange(schedule, N)
        
        # Compute divergence
        div_h = compute_divergence_upwind(h_all, Vx_all, Vy_all, Vz_all, geometry, halo_fn)
        
        # Statistics
        div_abs = jnp.abs(div_h)
        
        print(f"  div range: [{float(div_h.min()):.2e}, {float(div_h.max()):.2e}]")
        print(f"  |div| max: {float(div_abs.max()):.2e}")
        print(f"  |div| mean: {float(div_abs.mean()):.2e}")
        
        # Check edges vs interior
        interior_max = 0.0
        edge_max = 0.0
        corner_max = 0.0
        
        for face in range(6):
            for i in range(N):
                for j in range(N):
                    val = abs(float(div_h[face, i, j]))
                    is_edge_i = (i == 0 or i == N-1)
                    is_edge_j = (j == 0 or j == N-1)
                    
                    if is_edge_i and is_edge_j:
                        corner_max = max(corner_max, val)
                    elif is_edge_i or is_edge_j:
                        edge_max = max(edge_max, val)
                    else:
                        interior_max = max(interior_max, val)
        
        print(f"  Interior max |div|: {interior_max:.2e}")
        print(f"  Edge max |div|: {edge_max:.2e}  (but not corners)")
        print(f"  Corner max |div|: {corner_max:.2e}  (the 4 corner cells)")
        
        # Show where the max edge errors are located
        for face in range(6):
            face_div = np.abs(np.array(div_h[face]))
            # Check each edge
            top_max = face_div[-1, 1:-1].max()
            bot_max = face_div[0, 1:-1].max()
            left_max = face_div[1:-1, 0].max()
            right_max = face_div[1:-1, -1].max()
            max_edge_val = max(top_max, bot_max, left_max, right_max)
            if max_edge_val > 1e-8:
                print(f"    Face {face}: top={top_max:.2e} bot={bot_max:.2e} left={left_max:.2e} right={right_max:.2e}")
        
        # Global integral of divergence (should be machine precision for conservation)
        R = 6.371e6
        sqrtG_all = jnp.array(geometry.sqrtG) * R**2
        global_div = 0.0
        for face in range(6):
            global_div += float(jnp.sum(div_h[face] * sqrtG_all[face]) * geometry.dx**2)
        
        total_mass = float(jnp.sum(h_all * sqrtG_all) * geometry.dx**2)
        rel_div = abs(global_div) / total_mass
        
        print(f"  Global integral of div: {global_div:.2e}")
        print(f"  Relative (conservation): {rel_div:.2e}")
        
        if rel_div < 1e-12:
            print("  ✓ Global conservation is machine precision")
        elif div_abs.max() < 1e-6:
            print("  ✓ Local divergence is small")
        elif div_abs.max() < 1e-3:
            print("  ~ Local divergence is moderate")
        else:
            print("  ✗ Local divergence is large - problem!")


if __name__ == "__main__":
    main()

