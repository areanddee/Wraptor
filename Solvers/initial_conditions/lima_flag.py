"""
Lima Flag Initial Condition

Checkerboard pattern on Face 0 (north pole) for testing thermal diffusion
and demonstrating cross-face connectivity.

Pattern:
    Face 0: Upper-left & lower-right quadrants = T_hot
            Upper-right & lower-left quadrants = T_background
    Faces 1-5: T_background
"""

import jax.numpy as jnp
import numpy as np
from typing import Union

from Solvers.geometry import CubedSphereGeometry


def lima_flag(geometry: CubedSphereGeometry,
              T_hot: float = 600.0,
              T_background: float = 1.0,
              dtype=None) -> jnp.ndarray:
    """
    Create Lima Flag initial condition for thermal diffusion.
    
    This pattern places hot quadrants on Face 0 (north pole) in a 
    checkerboard arrangement. Heat diffuses across face boundaries,
    demonstrating correct halo exchange.
    
    Args:
        geometry: CubedSphereGeometry instance
        T_hot: Temperature of hot quadrants [K]
        T_background: Background temperature [K] (avoid 0 for log plots)
        dtype: Output dtype (default: float64)
        
    Returns:
        T: (6, N, N) temperature field
        
    Note:
        - Uses T_background=1.0 by default to avoid log(0) in visualization
        - Geometry is used only for dimensions (pattern is grid-based)
    """
    if dtype is None:
        dtype = jnp.float64
    
    N = geometry.N
    n_faces = geometry.n_faces
    
    # Initialize to background
    T = jnp.ones((n_faces, N, N), dtype=dtype) * T_background
    
    half_N = N // 2
    
    # Upper-left quadrant: hot
    T = T.at[0, :half_N, :half_N].set(T_hot)
    
    # Lower-right quadrant: hot  
    T = T.at[0, half_N:, half_N:].set(T_hot)
    
    return T


def lima_flag_description() -> str:
    """Return description for logging/metadata."""
    return """Lima Flag Initial Condition:
    Face 0 (north pole): Checkerboard pattern
      - Upper-left quadrant: T_hot (600K default)
      - Lower-right quadrant: T_hot
      - Other quadrants: T_background (1K default)
    Faces 1-5: T_background
    
    Purpose: Test thermal diffusion and cross-face halo exchange."""

