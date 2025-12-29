"""
2-Deep Halo Exchange for Cubed-Sphere Grids

Required for Putman & Lin (2007) 3rd-order edge extrapolation which needs
q₁ and q₂ (first two cells) from the neighboring face.

This is a TRUE 2-deep exchange (not 1-deep applied twice):
- Extracts 2 rows of interior cells from each face edge
- Exchanges both rows in a single operation
- Inserts as 2 ghost cell layers

Based on the connectivity specification in:
  Solvers/halo_exchange.py
"""

import jax
import jax.numpy as jnp
from functools import partial

from Solvers.halo_exchange import create_communication_schedule


def extend_to_include_ghosts_2deep(field_interior, N):
    """
    Extend (6, N, N) interior-only to (6, N+4, N+4) with 2 ghost cells per edge.
    
    Args:
        field_interior: (6, N, N) interior cells
        N: Grid resolution
    
    Returns:
        field_ghosts: (6, N+4, N+4) with 2-cell ghost regions initialized to edge values
    """
    # Pad with edge values (will be overwritten by halo exchange)
    return jnp.pad(field_interior, ((0, 0), (2, 2), (2, 2)), mode='edge')


def extract_edge_2deep(field_face, edge, N):
    """
    Extract 2 rows/columns of cells adjacent to an edge.
    
    IMPORTANT: Must match 1-deep convention from Solvers/halo_exchange.py:
    - E/W edges use first array index (rows)
    - N/S edges use second array index (columns)
    
    Args:
        field_face: (N+4, N+4) field with 2-deep ghost cells
        edge: 'N', 'S', 'E', 'W'
        N: Interior grid size
    
    Returns:
        row1, row2: Two arrays of shape (N,) - first and second cell rows from edge
    """
    # Interior region is [2:N+2, 2:N+2] in the (N+4, N+4) array
    interior = field_face[2:N+2, 2:N+2]
    
    if edge == 'E':
        # East edge: last row (i = N-1), uses FIRST index
        return interior[-1, :], interior[-2, :]
    elif edge == 'W':
        # West edge: first row (i = 0), uses FIRST index
        return interior[0, :], interior[1, :]
    elif edge == 'N':
        # North edge: last column (j = N-1), uses SECOND index
        return interior[:, -1], interior[:, -2]
    elif edge == 'S':
        # South edge: first column (j = 0), uses SECOND index
        return interior[:, 0], interior[:, 1]
    else:
        raise ValueError(f"Unknown edge: {edge}")


def insert_ghost_2deep(field_face, edge, ghost1, ghost2, N):
    """
    Insert 2 rows/columns of ghost cells at an edge.
    
    IMPORTANT: Must match 1-deep convention from Solvers/halo_exchange.py:
    - E/W edges use first array index (rows)
    - N/S edges use second array index (columns)
    
    Args:
        field_face: (N+4, N+4) field with ghost cells
        edge: 'N', 'S', 'E', 'W'
        ghost1, ghost2: Two arrays of shape (N,) - ghost cell values
                        ghost1 = inner ghost (adjacent to interior)
                        ghost2 = outer ghost (further from interior)
        N: Interior grid size
    
    Returns:
        Updated field_face with ghost values inserted
    """
    if edge == 'E':
        # East ghost rows: rows N+2 and N+3 (uses FIRST index)
        field_face = field_face.at[N+2, 2:N+2].set(ghost1)
        field_face = field_face.at[N+3, 2:N+2].set(ghost2)
    elif edge == 'W':
        # West ghost rows: rows 0 and 1 (uses FIRST index)
        field_face = field_face.at[1, 2:N+2].set(ghost1)
        field_face = field_face.at[0, 2:N+2].set(ghost2)
    elif edge == 'N':
        # North ghost columns: columns N+2 and N+3 (uses SECOND index)
        field_face = field_face.at[2:N+2, N+2].set(ghost1)
        field_face = field_face.at[2:N+2, N+3].set(ghost2)
    elif edge == 'S':
        # South ghost columns: columns 0 and 1 (uses SECOND index)
        field_face = field_face.at[2:N+2, 1].set(ghost1)
        field_face = field_face.at[2:N+2, 0].set(ghost2)
    
    return field_face


def apply_operation_2deep(data1, data2, operation):
    """
    Apply rotation/transpose operations to a pair of data rows.
    
    Must match the logic in Solvers/halo_exchange.py:apply_operations()
    
    Args:
        data1, data2: Arrays of shape (N,) to transform
        operation: 'N' (none), 'R' (reverse), 'T' (transpose), 'TR' (both)
    
    Returns:
        Transformed (data1, data2)
    """
    if operation == 'N':
        return data1, data2
    elif operation == 'R':
        return data1[::-1], data2[::-1]
    elif operation == 'T':
        # For 1D edge data, transpose is identity
        return data1, data2
    elif operation == 'TR':
        # Transpose + reverse = just reverse for 1D data
        return data1[::-1], data2[::-1]
    else:
        raise ValueError(f"Unknown operation: {operation}")


def exchange_pair_2deep(field, face_a, edge_a, face_b, edge_b, operation, N):
    """
    Exchange 2-deep halo data between two faces.
    
    Args:
        field: (6, N+4, N+4) field with ghost cells
        face_a, face_b: Face indices (0-5)
        edge_a, edge_b: Edge identifiers ('N', 'S', 'E', 'W')
        operation: Transformation to apply ('N', 'R', 'T', 'TR')
        N: Interior grid size
    
    Returns:
        Updated field with exchanged ghost cells
    """
    # Extract 2 rows from each face's edge
    a_row1, a_row2 = extract_edge_2deep(field[face_a], edge_a, N)
    b_row1, b_row2 = extract_edge_2deep(field[face_b], edge_b, N)
    
    # Apply operations
    a_to_b_1, a_to_b_2 = apply_operation_2deep(a_row1, a_row2, operation)
    b_to_a_1, b_to_a_2 = apply_operation_2deep(b_row1, b_row2, operation)
    
    # Insert as ghost cells on the opposite face
    field = field.at[face_b].set(
        insert_ghost_2deep(field[face_b], edge_b, a_to_b_1, a_to_b_2, N)
    )
    field = field.at[face_a].set(
        insert_ghost_2deep(field[face_a], edge_a, b_to_a_1, b_to_a_2, N)
    )
    
    return field


def make_halo_exchange_2deep(schedule, N):
    """
    Create a JIT-compiled 2-deep halo exchange function.
    
    Args:
        schedule: Communication schedule from create_communication_schedule()
        N: Interior grid size
    
    Returns:
        JIT-compiled function: field_ghosts -> field_ghosts with exchanged halos
    """
    def exchange_all(field_ghosts):
        """Exchange all 12 edges in 4 stages."""
        field = field_ghosts
        
        for stage in schedule:
            for (face_a, edge_a), (face_b, edge_b), operation in stage:
                field = exchange_pair_2deep(
                    field, face_a, edge_a, face_b, edge_b, operation, N
                )
        
        return field
    
    return jax.jit(exchange_all)


def exchange_scalar_halos_2deep(field_interior, N, halo_exchange_fn):
    """
    Convenience function: extend + exchange for 2-deep halos.
    
    Args:
        field_interior: (6, N, N) interior field
        N: Grid resolution
        halo_exchange_fn: JIT-compiled exchange function from make_halo_exchange_2deep
    
    Returns:
        field_ghosts: (6, N+4, N+4) with exchanged ghost cells
    """
    field_ghosts = extend_to_include_ghosts_2deep(field_interior, N)
    return halo_exchange_fn(field_ghosts)


# ============================================================================
# EDGE DATA EXTRACTION FOR PUTMAN & LIN
# ============================================================================

def get_edge_neighbor_data_2deep(field_ghosts, face_id, N):
    """
    Extract q₁ and q₂ from all 4 edges of a face for Putman & Lin edge extrapolation.
    
    Args:
        field_ghosts: (6, N+4, N+4) field with 2-deep ghost cells (already exchanged)
        face_id: Current face index (0-5)
        N: Interior grid size
    
    Returns:
        dict with keys 'left', 'right', 'bottom', 'top'
        Each contains (q1_neighbor, q2_neighbor) where:
            q1_neighbor = first ghost cell row/col (from neighbor's edge)
            q2_neighbor = second ghost cell row/col (from neighbor's interior)
    
    Note: West/East (left/right) use FIRST index (rows)
          North/South (top/bottom) use SECOND index (columns)
          This matches the convention in halo_exchange.py
    """
    f = field_ghosts[face_id]
    
    return {
        # West (left): ghost ROWS at i=0,1
        'left': (f[1, 2:N+2], f[0, 2:N+2]),
        
        # East (right): ghost ROWS at i=N+2, N+3
        'right': (f[N+2, 2:N+2], f[N+3, 2:N+2]),
        
        # South (bottom): ghost COLUMNS at j=0,1
        'bottom': (f[2:N+2, 1], f[2:N+2, 0]),
        
        # North (top): ghost COLUMNS at j=N+2, N+3
        'top': (f[2:N+2, N+2], f[2:N+2, N+3]),
    }


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_2deep_halo():
    """
    Test 2-deep halo exchange.
    
    This is a TRUE 2-deep exchange in a single pass:
    - Each edge sends 2 interior rows (q₁, q₂) to neighbor
    - Neighbor receives them as 2 ghost cell layers
    - Required for Putman & Lin Eq 47: q_e = [7(q₁ʳ + q₁ˡ) - (q₂ʳ + q₂ˡ)] / 12
    """
    print("=" * 60)
    print("Testing 2-deep halo exchange")
    print("=" * 60)
    
    N = 10
    schedule = create_communication_schedule()
    halo_fn = make_halo_exchange_2deep(schedule, N)
    
    # Test 1: Constant field (verifies connectivity)
    print("\nTest 1: Constant field per face (verifies connectivity)")
    print("-" * 40)
    
    field = jnp.zeros((6, N, N))
    for face in range(6):
        field = field.at[face].set(face)
    
    field_ghosts = exchange_scalar_halos_2deep(field, N, halo_fn)
    
    print(f"  Input shape: (6, {N}, {N})")
    print(f"  Output shape: {field_ghosts.shape}")
    
    for face in range(6):
        edge_data = get_edge_neighbor_data_2deep(field_ghosts, face, N)
        print(f"\n  Face {face}:")
        for edge, (q1, q2) in edge_data.items():
            print(f"    {edge}: q1={float(q1[0]):.0f}, q2={float(q2[0]):.0f}")
    
    # Test 2: Gradient field (verifies q1 ≠ q2)
    print("\n" + "=" * 60)
    print("Test 2: Gradient field (verifies q1 ≠ q2)")
    print("-" * 40)
    
    # Create field with gradient: value = face*100 + row_index
    field_grad = jnp.zeros((6, N, N))
    for face in range(6):
        for i in range(N):
            field_grad = field_grad.at[face, i, :].set(face * 100 + i)
    
    field_ghosts_grad = exchange_scalar_halos_2deep(field_grad, N, halo_fn)
    
    print(f"  Field values: face*100 + row_index")
    print(f"  So face 0 has rows [0, 1, 2, ..., 9]")
    print(f"  And face 1 has rows [100, 101, 102, ..., 109]")
    
    # Check Face 0's top edge (should get from Face 1)
    edge_data = get_edge_neighbor_data_2deep(field_ghosts_grad, 0, N)
    q1_top, q2_top = edge_data['top']
    print(f"\n  Face 0, top edge (neighbor is Face 1):")
    print(f"    q1 = {float(q1_top[0]):.0f} (should be Face1's row N-1 = 109)")
    print(f"    q2 = {float(q2_top[0]):.0f} (should be Face1's row N-2 = 108)")
    
    if abs(float(q1_top[0]) - 109) < 1 and abs(float(q2_top[0]) - 108) < 1:
        print("    ✓ Correct: q1 and q2 are different rows from neighbor!")
    else:
        print("    ✗ Error: q1 and q2 values unexpected")
    
    print("\n" + "=" * 60)
    print("✓ 2-deep halo exchange tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_2deep_halo()

