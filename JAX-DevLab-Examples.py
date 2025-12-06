"""
FV Cubed-Sphere Shallow Water Solver - Code Snippets
Copyright © 2025 [Your Name]. All Rights Reserved.

PRELIMINARY RESEARCH CODE - PUBLICATION PENDING
This code is provided for review and educational purposes only
at the Fall 2025 Jax DevLab Conference, November 18-19, 2025. 
Not for redistribution, modification, or use without explicit written permission.

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.

Contact: richard.d.loft@areanddee.com
"""

def setup_sharding(self):
        """Setup JAX sharding for multi-device execution."""
        para_config = self.config['parallelization']
        device_type = para_config.get('device_type', 'cpu')
        num_devices = para_config.get('num_devices', 6)
        tiles_per_edge = para_config.get('tiles_per_edge', 1)
        
        print(f"\n{'='*70}")
        print(f"SETTING UP JAX SHARDING")
        print(f"{'='*70}")
        
        # Validate tiles_per_edge
        if tiles_per_edge != 1:
            raise NotImplementedError(
                f"Error: tiles_per_edge = {tiles_per_edge} is not yet supported.\n"
                f"Currently only tiles_per_edge = 1 is implemented (6 tiles total).\n"
                f"Future work will enable multi-tile-per-face for scaling to 100+ GPUs.\n"
                f"Example: tiles_per_edge = 3 → 54 tiles → run on 56 B200 chips."
            )
        
        # Calculate total tiles
        num_tiles = 6 * tiles_per_edge * tiles_per_edge
        
        # Validate device count
        if num_devices > num_tiles:
            raise ValueError(
                f"Error: num_devices = {num_devices} exceeds num_tiles = {num_tiles}.\n"
                f"Cannot have more devices than tiles.\n"
                f"With tiles_per_edge = {tiles_per_edge}, max devices = {num_tiles}."
            )
        
        if num_tiles % num_devices != 0:
            valid_counts = [d for d in range(1, num_tiles + 1) if num_tiles % d == 0]
            raise ValueError(
                f"Error: num_tiles = {num_tiles} is not evenly divisible by num_devices = {num_devices}.\n"
                f"JAX requires: num_tiles % num_devices == 0\n"
                f"With tiles_per_edge = {tiles_per_edge}, valid device counts are:\n"
                f"  {valid_counts}"
            )
        
        print(f"  Tile configuration:")
        print(f"    tiles_per_edge: {tiles_per_edge}")
        print(f"    total tiles: {num_tiles} (6 faces × {tiles_per_edge}² tiles/face)")
        print(f"    tiles per device: {num_tiles / num_devices:.1f}")
        
        if device_type == 'cpu':
            # Create virtual devices for CPU testing
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'
            print(f"  Device type: CPU (virtual devices)")
            print(f"  XLA_FLAGS set: {num_devices} virtual CPU devices")
        else:
            print(f"  Device type: GPU")
            print(f"  Requested devices: {num_devices}")
        
        devices = jax.devices()[:num_devices]
        print(f"  Available devices: {len(jax.devices())}")
        print(f"  Using devices: {devices}")
        
        # Create mesh: 1D array of devices mapped to 'tiles' axis
        self.mesh = Mesh(devices, ('tiles',))
        self.sharding = NamedSharding(self.mesh, P('tiles'))
        
        print(f"  Mesh created: {num_tiles} tiles across {len(devices)} device(s)")
        print(f"  Sharding strategy: PartitionSpec('tiles') on axis 0")
        if num_tiles > len(devices):
            print(f"  Note: Multiple tiles per device (serial execution)")
        print(f"{'='*70}\n")



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

    return field_with_ghosts[:, 1:N+1, 1:N+1]

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

