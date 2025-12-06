"""
Cosine Bell Initial Condition

Williamson Test Case 1: A smooth cosine bell for advection testing.
The bell is advected around the sphere by solid-body rotation.

Reference: Williamson, D.L., et al. (1992). 
"A standard test set for numerical approximations to the shallow 
water equations in spherical geometry." J. Comp. Phys.
"""

import jax.numpy as jnp
import numpy as np

from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams


def cosine_bell(geometry: CubedSphereGeometry,
                planet: PlanetParams,
                center_lat: float = 0.0,
                center_lon: float = -np.pi/2,  # 90°W
                radius_fraction: float = 1/3,
                amplitude: float = 1000.0,
                background: float = 0.0,
                dtype=None) -> jnp.ndarray:
    """
    Create cosine bell initial condition for advection.
    
    A smooth, compact tracer field centered at (center_lat, center_lon)
    with radius = radius_fraction * R_sphere.
    
    Args:
        geometry: CubedSphereGeometry instance
        planet: PlanetParams (for R_sphere)
        center_lat: Bell center latitude [radians]
        center_lon: Bell center longitude [radians]
        radius_fraction: Bell radius as fraction of planet radius
        amplitude: Maximum value at bell center
        background: Background value outside bell
        dtype: Output dtype (default: float64)
        
    Returns:
        q: (6, N, N) tracer concentration field
    """
    if dtype is None:
        dtype = jnp.float64
    
    N = geometry.N
    n_faces = geometry.n_faces
    R = planet.R_sphere
    
    # Bell radius
    R_bell = radius_fraction * R
    
    # Get lat/lon for all grid points
    lat, lon = geometry.get_lat_lon_all_faces()
    
    # Great circle distance from bell center
    # d = R * arccos(sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
    cos_angle = (np.sin(center_lat) * np.sin(lat) + 
                 np.cos(center_lat) * np.cos(lat) * np.cos(lon - center_lon))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety
    distance = R * np.arccos(cos_angle)
    
    # Cosine bell profile
    q = np.where(
        distance < R_bell,
        amplitude * 0.5 * (1.0 + np.cos(np.pi * distance / R_bell)),
        background
    )
    
    return jnp.array(q, dtype=dtype)


def cosine_bell_description() -> str:
    """Return description for logging/metadata."""
    return """Cosine Bell Initial Condition (Williamson Test Case 1):
    A smooth, axisymmetric tracer centered at specified lat/lon.
    Profile: q = A * 0.5 * (1 + cos(π * r / R_bell)) for r < R_bell
    
    Default: Center at equator, 90°W, radius = R_sphere/3
    
    Purpose: Test advection accuracy, shape preservation, mass conservation."""

