"""
3rd-Order Edge Extrapolation for Cubed-Sphere Grids

Based on Putman & Lin (2007): "Finite-volume transport on various cubed-sphere grids"
Journal of Computational Physics 227 (2007) 55-78

At cube edges where two faces meet, standard PLR reconstruction uses ghost cell
values that are only 1st-order accurate. This module implements the 3rd-order
directionally symmetric extrapolation from Putman & Lin that maintains accuracy.

Key equations:
- Eq 47: Shared edge value from both sides
    q_e = [7(q₁ʳ + q₁ˡ) - (q₂ʳ + q₂ˡ)] / 12

- Eq 49: Edge cell reconstruction (cubic polynomial fit)
    q_{e+} = (3q₁ + 11q₂ - 2m₂) / 14
    
where:
- q₁ʳ, q₁ˡ = first cell on right/left side of edge
- q₂ʳ, q₂ˡ = second cell on right/left side
- m₂ = PLR slope at second cell
"""

import jax.numpy as jnp
from typing import Tuple, Dict
import sys
from pathlib import Path

# Import 2-deep halo exchange
from halo_exchange_2deep import (
    create_communication_schedule,
    make_halo_exchange_2deep,
    exchange_scalar_halos_2deep,
    get_edge_neighbor_data_2deep
)


def compute_shared_edge_value(q1_this: jnp.ndarray, q1_neighbor: jnp.ndarray,
                              q2_this: jnp.ndarray, q2_neighbor: jnp.ndarray) -> jnp.ndarray:
    """
    Compute shared edge value using Eq 47 from Putman & Lin 2007.
    
    This is a 3rd-order accurate value at the interface between two cubed-sphere faces.
    
    Args:
        q1_this: First cell value on current face side of edge (N,)
        q1_neighbor: First cell value on neighbor face side of edge
        q2_this: Second cell value on current face side
        q2_neighbor: Second cell value on neighbor face side
    
    Returns:
        q_edge: Shared edge value with same shape as inputs
    """
    return (7.0 * (q1_this + q1_neighbor) - (q2_this + q2_neighbor)) / 12.0


def minmod(a, b):
    """Minmod limiter for two arguments."""
    return jnp.where(
        a * b > 0,
        jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)),
        0.0
    )


def minmod3(a, b, c):
    """Minmod limiter for three arguments."""
    return minmod(a, minmod(b, c))


def mc_limiter(central, forward, backward):
    """MC (Monotonized Central) limiter."""
    return minmod3(central, 2.0 * forward, 2.0 * backward)


def compute_limited_slopes_interior(field_ghosts: jnp.ndarray, dx: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute MC-limited slopes for interior cells only.
    
    This is the standard FV-PLR slope computation.
    
    Args:
        field_ghosts: (N+2, N+2) field with 1-deep ghost cells
        dx: Grid spacing [radians]
    
    Returns:
        slope1, slope2: (N, N) limited slopes
    """
    f = field_ghosts[1:-1, 1:-1]
    
    # ξ¹ direction
    f_im1 = field_ghosts[0:-2, 1:-1]
    f_ip1 = field_ghosts[2:, 1:-1]
    
    slope1_central = (f_ip1 - f_im1) / (2.0 * dx)
    slope1_forward = (f_ip1 - f) / dx
    slope1_backward = (f - f_im1) / dx
    slope1 = mc_limiter(slope1_central, slope1_forward, slope1_backward)
    
    # ξ² direction
    f_jm1 = field_ghosts[1:-1, 0:-2]
    f_jp1 = field_ghosts[1:-1, 2:]
    
    slope2_central = (f_jp1 - f_jm1) / (2.0 * dx)
    slope2_forward = (f_jp1 - f) / dx
    slope2_backward = (f - f_jm1) / dx
    slope2 = mc_limiter(slope2_central, slope2_forward, slope2_backward)
    
    return slope1, slope2


def compute_edge_aware_slopes(
    field_2deep: jnp.ndarray,
    face_id: int,
    dx: float,
    N: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute slopes with Putman & Lin 3rd-order edge extrapolation.
    
    For interior cells: Standard MC limiter
    For edge cells: Use Eq 47 to compute shared edge value, then derive slope
    
    Args:
        field_2deep: (6, N+4, N+4) field with 2-deep ghost cells (all faces, already exchanged)
        face_id: Current face index (0-5)
        dx: Grid spacing [radians]
        N: Interior grid size
    
    Returns:
        slope1, slope2: (N, N) slopes with edge extrapolation
    """
    # Extract this face's field with 1-deep padding for interior slope computation
    # Interior is at [2:N+2, 2:N+2] in the (N+4, N+4) array
    # We need [1:N+3, 1:N+3] for standard slope computation with 1-deep ghosts
    f_1deep = field_2deep[face_id, 1:N+3, 1:N+3]  # (N+2, N+2)
    
    # Compute standard interior slopes
    slope1, slope2 = compute_limited_slopes_interior(f_1deep, dx)
    
    # Get edge neighbor data from 2-deep exchange
    edge_data = get_edge_neighbor_data_2deep(field_2deep, face_id, N)
    
    # Interior cell values
    f_interior = field_2deep[face_id, 2:N+2, 2:N+2]  # (N, N)
    
    # IMPORTANT: Edge-slope mapping:
    # - Left/Right edges (j=0, j=N-1) → modify slope2 (j-direction crosses these edges)
    # - Bottom/Top edges (i=0, i=N-1) → modify slope1 (i-direction crosses these edges)
    
    # === Left edge (j=0): modify slope2[:, 0] ===
    q1_neighbor, q2_neighbor = edge_data['left']
    q1_this = f_interior[:, 0]      # First column on this side
    q2_this = f_interior[:, 1]      # Second column on this side
    
    # Shared edge value (Eq 47): 7*(q1_this + q1_neighbor) - (q2_this + q2_neighbor)
    q_edge_left = compute_shared_edge_value(q1_this, q1_neighbor, q2_this, q2_neighbor)
    
    # Slope at edge cell: value at left interface is q_edge, at center is q1_this
    # For PLR: q_left = q_center - 0.5*dx*slope → slope = (q_center - q_left) / (0.5*dx)
    edge_slope_left = (q1_this - q_edge_left) / (0.5 * dx)
    slope2 = slope2.at[:, 0].set(edge_slope_left)
    
    # === Right edge (j=N-1): modify slope2[:, -1] ===
    q1_neighbor, q2_neighbor = edge_data['right']
    q1_this = f_interior[:, -1]
    q2_this = f_interior[:, -2]
    
    # Shared edge value (Eq 47): 7*(q1_this + q1_neighbor) - (q2_this + q2_neighbor)
    q_edge_right = compute_shared_edge_value(q1_this, q1_neighbor, q2_this, q2_neighbor)
    
    # For PLR: q_right = q_center + 0.5*dx*slope → slope = (q_right - q_center) / (0.5*dx)
    edge_slope_right = (q_edge_right - q1_this) / (0.5 * dx)
    slope2 = slope2.at[:, -1].set(edge_slope_right)
    
    # === Bottom edge (i=0): modify slope1[0, :] ===
    q1_neighbor, q2_neighbor = edge_data['bottom']
    q1_this = f_interior[0, :]      # First row on this side
    q2_this = f_interior[1, :]      # Second row on this side
    
    # Shared edge value (Eq 47): 7*(q1_this + q1_neighbor) - (q2_this + q2_neighbor)
    q_edge_bottom = compute_shared_edge_value(q1_this, q1_neighbor, q2_this, q2_neighbor)
    edge_slope_bottom = (q1_this - q_edge_bottom) / (0.5 * dx)
    slope1 = slope1.at[0, :].set(edge_slope_bottom)
    
    # === Top edge (i=N-1): modify slope1[-1, :] ===
    q1_neighbor, q2_neighbor = edge_data['top']
    q1_this = f_interior[-1, :]
    q2_this = f_interior[-2, :]
    
    # Shared edge value (Eq 47): 7*(q1_this + q1_neighbor) - (q2_this + q2_neighbor)
    q_edge_top = compute_shared_edge_value(q1_this, q1_neighbor, q2_this, q2_neighbor)
    edge_slope_top = (q_edge_top - q1_this) / (0.5 * dx)
    slope1 = slope1.at[-1, :].set(edge_slope_top)
    
    return slope1, slope2


# ============================================================================
# PRESSURE GRADIENT WITH EDGE EXTRAPOLATION
# ============================================================================

def compute_pressure_gradient_edge_aware(
    h_2deep: jnp.ndarray,
    XI1_ghosts: jnp.ndarray,
    XI2_ghosts: jnp.ndarray,
    face_id: int,
    dx: float,
    N: int,
    g: float,
    R: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute pressure gradient -g∇h using FV-PLR with edge extrapolation.
    
    This uses Putman & Lin 2007 3rd-order edge handling for improved accuracy
    at cube edges.
    
    Args:
        h_2deep: (6, N+4, N+4) height field with 2-deep ghost cells (already exchanged)
        XI1_ghosts, XI2_ghosts: (N+2, N+2) coordinate arrays with 1-deep ghosts
        face_id: Current face index (0-5)
        dx: Grid spacing [rad]
        N: Interior grid size
        g: Gravity [m/s²]
        R: Planet radius [m]
    
    Returns:
        dVx_dt, dVy_dt, dVz_dt: Pressure gradient in Cartesian [m/s²]
    """
    # Compute edge-aware slopes
    h_slope1, h_slope2 = compute_edge_aware_slopes(h_2deep, face_id, dx, N)
    
    # Get interior coordinates
    xi1_int = XI1_ghosts[1:-1, 1:-1]
    xi2_int = XI2_ghosts[1:-1, 1:-1]
    
    # Import Jacobian computation from main solver
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Solvers"))
    from fv_plr_cubesphere_swe import compute_jacobian_face
    
    # Jacobian J: ∂(X,Y,Z)/∂(ξ¹,ξ²) has shape 3x2
    J11, J12, J21, J22, J31, J32 = compute_jacobian_face(xi1_int, xi2_int, face_id, R)
    
    # Gram matrix (J^T J)
    JTJ_11 = J11**2 + J21**2 + J31**2
    JTJ_12 = J11*J12 + J21*J22 + J31*J32
    JTJ_22 = J12**2 + J22**2 + J32**2
    det_JTJ = JTJ_11 * JTJ_22 - JTJ_12**2
    
    # (J^T J)^{-1}
    JTJi_11 = JTJ_22 / det_JTJ
    JTJi_12 = -JTJ_12 / det_JTJ
    JTJi_22 = JTJ_11 / det_JTJ
    
    # (J^T J)^{-1} @ [∂h/∂ξ¹, ∂h/∂ξ²]
    grad_xi_1 = JTJi_11 * h_slope1 + JTJi_12 * h_slope2
    grad_xi_2 = JTJi_12 * h_slope1 + JTJi_22 * h_slope2
    
    # J @ (above) gives Cartesian gradient
    dh_dX = J11 * grad_xi_1 + J12 * grad_xi_2
    dh_dY = J21 * grad_xi_1 + J22 * grad_xi_2
    dh_dZ = J31 * grad_xi_1 + J32 * grad_xi_2
    
    # Pressure gradient: -g ∇h
    return -g * dh_dX, -g * dh_dY, -g * dh_dZ


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_edge_extrapolation():
    """Test edge-aware slope computation."""
    import jax
    jax.config.update("jax_enable_x64", True)
    
    print("=" * 60)
    print("Testing Edge-Aware Slope Computation")
    print("=" * 60)
    
    N = 10
    dx = 0.1
    
    # Create halo exchange
    schedule = create_communication_schedule()
    halo_fn = make_halo_exchange_2deep(schedule, N)
    
    # Create test field: smooth function q = sin(i*dx) on each face
    import numpy as np
    field = jnp.zeros((6, N, N))
    for face in range(6):
        for i in range(N):
            for j in range(N):
                field = field.at[face, i, j].set(np.sin((i + face*10) * dx))
    
    # Exchange 2-deep halos
    field_2deep = exchange_scalar_halos_2deep(field, N, halo_fn)
    
    print(f"\n  Field shape: (6, {N}, {N})")
    print(f"  With 2-deep ghosts: {field_2deep.shape}")
    
    # Compute edge-aware slopes for face 0
    slope1, slope2 = compute_edge_aware_slopes(field_2deep, 0, dx, N)
    
    print(f"\n  Slopes for face 0:")
    print(f"    slope1 shape: {slope1.shape}")
    print(f"    slope1 range: [{float(slope1.min()):.4f}, {float(slope1.max()):.4f}]")
    print(f"    slope2 range: [{float(slope2.min()):.4f}, {float(slope2.max()):.4f}]")
    
    # Compare edge vs interior slopes
    # slope1 = i-direction slopes (modified at bottom/top edges, i=0 and i=N-1)
    # slope2 = j-direction slopes (modified at left/right edges, j=0 and j=N-1)
    print(f"\n  Edge vs Interior slopes:")
    print(f"    slope1 at bottom edge (i=0):   {float(slope1[0, N//2]):.4f}")
    print(f"    slope1 at interior (i=N/2):    {float(slope1[N//2, N//2]):.4f}")
    print(f"    slope1 at top edge (i=N-1):    {float(slope1[-1, N//2]):.4f}")
    print(f"    slope2 at left edge (j=0):     {float(slope2[N//2, 0]):.4f}")
    print(f"    slope2 at interior (j=N/2):    {float(slope2[N//2, N//2]):.4f}")
    print(f"    slope2 at right edge (j=N-1):  {float(slope2[N//2, -1]):.4f}")
    
    # The slopes should be cos(i*dx) * dx for sin(i*dx)
    expected_interior = np.cos((N//2) * dx)
    print(f"\n  Expected interior slope (cos): {expected_interior:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Edge-aware slope test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_edge_extrapolation()

