"""
Scalar Halo Exchange V2 - Optimized with functools.partial and JIT compilation

Key improvements for sharding and performance:
1. Static arguments frozen with functools.partial (face IDs, edges, N)
2. Each exchange function JIT-compiled once at initialization
3. No recompilation during timestepping
4. Composed function is also JIT-compiled for optimization

Based on CONNECTIVITY_SPECIFICATION.md with 12 buffer swaps in 4 stages.
"""

import jax
import jax.numpy as jnp
from functools import partial


def create_communication_schedule():
    """
    12 buffer swaps in 4 non-blocking stages.
    
    Format: ((face_a, edge_a), (face_b, edge_b), operations)
    
    Returns:
        Tuple of 4 stages, each containing 3 edge pairs
    """
    return (
        # Stage 0
        (
            ((0, "N"), (1, "N"), "R"),
            ((3, "E"), (4, "W"), "N"),
            ((2, "S"), (5, "E"), "TR")
        ),
        # Stage 1
        (
            ((0, "E"), (4, "N"), "T"),
            ((2, "E"), (3, "W"), "N"),
            ((1, "S"), (5, "N"), "N")
        ),
        # Stage 2
        (
            ((0, "W"), (2, "N"), "TR"),
            ((1, "W"), (4, "E"), "N"),
            ((3, "S"), (5, "S"), "R")
        ),
        # Stage 3
        (
            ((0, "S"), (3, "N"), "N"),
            ((1, "E"), (2, "W"), "N"),
            ((4, "S"), (5, "W"), "T")
        )
    )


def extend_to_include_ghosts(field_interior, N):
    """
    Extend (6, N, N) interior-only to (6, N+2, N+2) with ghost cells.
    
    Args:
        field_interior: (6, N, N) interior cells
        N: Interior resolution
        
    Returns:
        field_with_ghosts: (6, N+2, N+2) with zero-initialized ghosts
    """
    field_ghosts = jnp.zeros((6, N+2, N+2))
    
    # Copy interior to center [1:N+1, 1:N+1]
    for face in range(6):
        field_ghosts = field_ghosts.at[face, 1:N+1, 1:N+1].set(
            field_interior[face, :, :])
    
    return field_ghosts


def extract_interior(field_with_ghosts, N):
    """
    Extract (6, N, N) interior from (6, N+2, N+2).
    
    Args:
        field_with_ghosts: (6, N+2, N+2)
        N: Interior resolution
        
    Returns:
        field_interior: (6, N, N)
    """
    return field_with_ghosts[:, 1:N+1, 1:N+1]


def extract_boundary_data(face_data, edge, N):
    """
    Extract interior boundary data (N,) from face for sending.
    
    Args:
        face_data: (N+2, N+2) face with ghosts
        edge: "E", "W", "N", or "S"
        N: Interior resolution
        
    Returns:
        boundary_data: (N,) values from interior cells at boundary
    """
    if edge == "E":  # East: rightmost interior column (i=N)
        return face_data[N, 1:N+1]
    elif edge == "W":  # West: leftmost interior column (i=1)
        return face_data[1, 1:N+1]
    elif edge == "N":  # North: topmost interior row (j=N)
        return face_data[1:N+1, N]
    elif edge == "S":  # South: bottommost interior row (j=1)
        return face_data[1:N+1, 1]
    else:
        raise ValueError(f"Unknown edge: {edge}")


def set_ghost_data(face_data, edge, data, N):
    """
    Write data to ghost cells.
    
    Args:
        face_data: (N+2, N+2) face with ghosts
        edge: "E", "W", "N", or "S"
        data: (N,) values to write to ghost cells
        N: Interior resolution
        
    Returns:
        Updated face_data
    """
    if edge == "E":  # East ghost: i=N+1
        return face_data.at[N+1, 1:N+1].set(data)
    elif edge == "W":  # West ghost: i=0
        return face_data.at[0, 1:N+1].set(data)
    elif edge == "N":  # North ghost: j=N+1
        return face_data.at[1:N+1, N+1].set(data)
    elif edge == "S":  # South ghost: j=0
        return face_data.at[1:N+1, 0].set(data)
    else:
        raise ValueError(f"Unknown edge: {edge}")


def apply_operations(data, operations):
    """
    Apply transpose and/or reverse operations to boundary data.
    
    Args:
        data: (N,) boundary data
        operations: "N", "T", "R", or "TR"
        
    Returns:
        Transformed data (N,)
    """
    if operations == "N":
        return data
    elif operations == "T":
        return data
    elif operations == "R":
        return data[::-1]
    elif operations == "TR":
        return data[::-1]
    else:
        raise ValueError(f"Unknown operation: {operations}")


def exchange_edge_pair(field_ghosts, face_a, edge_a, face_b, edge_b, 
                       operations, N):
    """
    Bidirectional exchange between two edges.
    
    This is the core function that will be pre-compiled with static arguments.
    
    Args:
        field_ghosts: (6, N+2, N+2) with uninitialized ghosts
        face_a, edge_a: First edge (STATIC)
        face_b, edge_b: Second edge (STATIC)
        operations: "N", "T", "R", or "TR" (STATIC)
        N: Interior resolution (STATIC)
        
    Returns:
        Updated field_ghosts with filled ghost cells
    """
    # Extract boundary data from interior cells at edges
    data_a = extract_boundary_data(field_ghosts[face_a], edge_a, N)
    data_b = extract_boundary_data(field_ghosts[face_b], edge_b, N)
    
    # Apply operations
    data_to_b = apply_operations(data_a, operations)
    data_to_a = apply_operations(data_b, operations)
    
    # Write to ghost cells
    field_ghosts = field_ghosts.at[face_b].set(
        set_ghost_data(field_ghosts[face_b], edge_b, data_to_b, N))
    field_ghosts = field_ghosts.at[face_a].set(
        set_ghost_data(field_ghosts[face_a], edge_a, data_to_a, N))
    
    return field_ghosts


def make_halo_exchange(schedule, N):
    """
    Factory creates JIT-compiled exchange functions.
    
    This is the key optimization:
    1. Pre-compile each exchange with static arguments frozen
    2. Each exchange_fn compiles once at initialization
    3. No recompilation during timestepping
    4. Composed function is also JIT-compiled
    
    Args:
        schedule: Communication schedule from create_communication_schedule()
        N: Interior resolution (frozen as static argument)
        
    Returns:
        cubesphere_halo_exchange: JIT-compiled function that takes field_ghosts
    """
    exchange_functions = []
    
    print("Pre-compiling halo exchange functions...")
    
    # Pre-compile each exchange in schedule
    for stage_idx, stage in enumerate(schedule):
        for (face_a, edge_a), (face_b, edge_b), operations in stage:
            # Use partial to bake in static arguments
            exchange_fn = partial(
                exchange_edge_pair,
                face_a=face_a, edge_a=edge_a,
                face_b=face_b, edge_b=edge_b,
                operations=operations, N=N
            )
            
            # JIT compile once
            exchange_fn_jit = jax.jit(exchange_fn)
            exchange_functions.append(exchange_fn_jit)
            
            print(f"  Stage {stage_idx}: ({face_a},{edge_a}) ↔ ({face_b},{edge_b}) [{operations}]")
    
    # Return composed function that applies all exchanges
    def cubesphere_halo_exchange(field_ghosts):
        """Apply all pre-compiled exchanges."""
        for exchange_fn in exchange_functions:
            field_ghosts = exchange_fn(field_ghosts)
        return field_ghosts
    
    # JIT compile the composed function for additional optimization
    print("JIT compiling composed exchange function...")
    return jax.jit(cubesphere_halo_exchange)


def exchange_scalar_halos_v2(field_interior, N, halo_exchange_fn=None):
    """
    Convenience function: extend, exchange, extract.
    
    Args:
        field_interior: (6, N, N) interior cells only
        N: Interior resolution
        halo_exchange_fn: Pre-compiled exchange function (optional)
                         If None, creates a non-optimized version
        
    Returns:
        field_with_ghosts: (6, N+2, N+2) with filled ghosts
    """
    # Extend
    field_ghosts = extend_to_include_ghosts(field_interior, N)
    
    # Exchange
    if halo_exchange_fn is not None:
        # Use pre-compiled function (FAST)
        field_ghosts = halo_exchange_fn(field_ghosts)
    else:
        # Fallback to non-optimized version (SLOW - creates schedule every call)
        schedule = create_communication_schedule()
        for stage in schedule:
            for (face_a, edge_a), (face_b, edge_b), operations in stage:
                field_ghosts = exchange_edge_pair(
                    field_ghosts, face_a, edge_a, face_b, edge_b,
                    operations, N)
    
    return field_ghosts


# =============================================================================
# TESTING AND VERIFICATION
# =============================================================================

def test_constant_field_v2():
    """Test 1: Constant field with optimized version."""
    print("\nTest 1: Constant field (V2 - optimized)")
    print("-" * 60)
    
    N = 10
    constant_value = 42.0
    
    # Initialize constant field
    field = jnp.ones((6, N, N)) * constant_value
    
    # Create pre-compiled exchange function
    schedule = create_communication_schedule()
    halo_exchange_fn = make_halo_exchange(schedule, N)
    
    # Exchange (using pre-compiled function)
    print("\nApplying halo exchange...")
    field_ghosts = exchange_scalar_halos_v2(field, N, halo_exchange_fn)
    
    # Check: all edge ghosts should equal constant_value
    for face in range(6):
        assert jnp.allclose(field_ghosts[face, 0, 1:N+1], constant_value), \
            f"Face {face} West ghost failed"
        assert jnp.allclose(field_ghosts[face, N+1, 1:N+1], constant_value), \
            f"Face {face} East ghost failed"
        assert jnp.allclose(field_ghosts[face, 1:N+1, 0], constant_value), \
            f"Face {face} South ghost failed"
        assert jnp.allclose(field_ghosts[face, 1:N+1, N+1], constant_value), \
            f"Face {face} North ghost failed"
    
    print("✓ All ghost cells equal constant value")
    print(f"  Expected: {constant_value}")
    print(f"  Min ghost: {jnp.min(field_ghosts)}")
    print(f"  Max ghost: {jnp.max(field_ghosts)}")


def test_face_id_pattern_v2():
    """Test 2: Face ID pattern with optimized version."""
    print("\nTest 2: Face ID pattern (V2 - optimized)")
    print("-" * 60)
    
    N = 10
    
    # Initialize: each face has its face ID as value
    field = jnp.zeros((6, N, N))
    for face in range(6):
        field = field.at[face].set(float(face))
    
    # Create pre-compiled exchange function
    schedule = create_communication_schedule()
    halo_exchange_fn = make_halo_exchange(schedule, N)
    
    # Exchange
    print("\nApplying halo exchange...")
    field_ghosts = exchange_scalar_halos_v2(field, N, halo_exchange_fn)
    
    print("\nChecking sample connections:")
    
    # (3,E) ↔ (4,W) - Stage 0, no operations
    face3_east_ghost = field_ghosts[3, N+1, 1:N+1]
    expected = 4.0
    assert jnp.allclose(face3_east_ghost, expected)
    print(f"  ✓ (3,E) ↔ (4,W): Face 3 East ghost = {face3_east_ghost[0]} (expect 4.0)")
    
    face4_west_ghost = field_ghosts[4, 0, 1:N+1]
    expected = 3.0
    assert jnp.allclose(face4_west_ghost, expected)
    print(f"  ✓ (3,E) ↔ (4,W): Face 4 West ghost = {face4_west_ghost[0]} (expect 3.0)")
    
    print("\n✓ Connectivity verified for sample connections")


def test_performance_comparison():
    """Test 3: Performance comparison - optimized vs non-optimized."""
    import time
    
    print("\nTest 3: Performance comparison")
    print("-" * 60)
    
    N = 60
    field = jnp.ones((6, N, N)) * 100.0
    
    # Create pre-compiled version
    schedule = create_communication_schedule()
    print("\nCreating optimized version...")
    halo_exchange_fn = make_halo_exchange(schedule, N)
    
    # Warmup
    print("\nWarming up JIT compilation...")
    _ = exchange_scalar_halos_v2(field, N, halo_exchange_fn)
    
    # Benchmark optimized version
    print("\nBenchmarking optimized version (10 iterations)...")
    start = time.time()
    for _ in range(10):
        _ = exchange_scalar_halos_v2(field, N, halo_exchange_fn)
    optimized_time = (time.time() - start) / 10
    
    # Benchmark non-optimized version
    print("Benchmarking non-optimized version (10 iterations)...")
    start = time.time()
    for _ in range(10):
        _ = exchange_scalar_halos_v2(field, N, None)  # No pre-compiled function
    non_optimized_time = (time.time() - start) / 10
    
    print(f"\nResults:")
    print(f"  Optimized:     {optimized_time*1000:.3f} ms/iteration")
    print(f"  Non-optimized: {non_optimized_time*1000:.3f} ms/iteration")
    print(f"  Speedup:       {non_optimized_time/optimized_time:.2f}x")
    
    # Verify correctness
    result_opt = exchange_scalar_halos_v2(field, N, halo_exchange_fn)
    result_non_opt = exchange_scalar_halos_v2(field, N, None)
    
    assert jnp.allclose(result_opt, result_non_opt), "Results don't match!"
    print("\n✓ Both methods produce identical results")


if __name__ == "__main__":
    print("=" * 60)
    print("SCALAR HALO EXCHANGE V2 - OPTIMIZED WITH PARTIAL + JIT")
    print("=" * 60)
    
    try:
        test_constant_field_v2()
        test_face_id_pattern_v2()
        test_performance_comparison()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nKey optimization benefits:")
        print("  • Static arguments frozen with functools.partial")
        print("  • Each exchange function JIT-compiled once")
        print("  • No recompilation during timestepping")
        print("  • Ready for shard_map deployment")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise

