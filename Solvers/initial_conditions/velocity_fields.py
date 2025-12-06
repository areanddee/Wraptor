"""
Velocity Field Initial Conditions

Analytic velocity fields for advection testing.
All fields are in Cartesian (X, Y, Z) components on the unit sphere,
scaled by planet radius as needed.
"""

import jax.numpy as jnp
import numpy as np

from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams


def solid_body_rotation(geometry: CubedSphereGeometry,
                        planet: PlanetParams,
                        rotation_period_days: float = 12.0,
                        alpha: float = 0.0,
                        dtype=None) -> tuple:
    """
    Solid body rotation velocity field.
    
    Creates a velocity field corresponding to solid-body rotation
    around an axis tilted by angle alpha from the polar axis.
    
    For alpha=0: Rotation around Z-axis (like Earth's rotation)
    For alpha=π/2 - 0.05: Tilted rotation (Williamson standard)
    
    Args:
        geometry: CubedSphereGeometry instance
        planet: PlanetParams (for R_sphere and default omega)
        rotation_period_days: Period of one full rotation [days]
        alpha: Tilt angle of rotation axis [radians]
        dtype: Output dtype (default: float64)
        
    Returns:
        Vx, Vy, Vz: (6, N, N) Cartesian velocity components [m/s]
    """
    if dtype is None:
        dtype = jnp.float64
    
    R = planet.R_sphere
    
    # Angular velocity for specified rotation period
    # u0 = equatorial velocity for 12-day rotation
    seconds_per_day = 86400.0
    omega_rotation = 2 * np.pi / (rotation_period_days * seconds_per_day)
    u0 = omega_rotation * R  # Maximum velocity at equator
    
    # Get Cartesian coordinates on unit sphere
    X, Y, Z = geometry.get_xyz_all_faces()
    
    # Solid body rotation: v = ω × r
    # For rotation around Z-axis: v = ω * (-Y, X, 0) in direction
    # But we need to account for tilt angle alpha
    
    # Rotation axis: (sin(alpha), 0, cos(alpha)) for tilt in X-Z plane
    # v = ω × r = ω * |axis × position|
    
    if alpha == 0.0:
        # Simple rotation around Z-axis
        # v = u0 * (-Y/R, X/R, 0) at equator, scaled by cos(lat)
        # In 3D: v = omega * R * (-sin(lon)*cos(lat), cos(lon)*cos(lat), 0)
        # Simplified for unit sphere then scaled:
        Vx = -u0 * Y  # -Y on unit sphere
        Vy = u0 * X   # X on unit sphere
        Vz = np.zeros_like(X)
    else:
        # Tilted rotation (Williamson standard: alpha = π/2 - 0.05)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        
        # Rotation axis: n = (sin(alpha), 0, cos(alpha))
        # v = u0 * (n × r_hat) where r_hat = (X, Y, Z) on unit sphere
        # n × r = (0*Z - cos_alpha*Y, cos_alpha*X - sin_alpha*Z, sin_alpha*Y - 0*X)
        #       = (-cos_alpha*Y, cos_alpha*X - sin_alpha*Z, sin_alpha*Y)
        
        Vx = u0 * (-cos_alpha * Y)
        Vy = u0 * (cos_alpha * X - sin_alpha * Z)
        Vz = u0 * (sin_alpha * Y)
    
    return (jnp.array(Vx, dtype=dtype), 
            jnp.array(Vy, dtype=dtype), 
            jnp.array(Vz, dtype=dtype))


def solid_body_rotation_description() -> str:
    """Return description for logging/metadata."""
    return """Solid Body Rotation Velocity Field:
    Uniform angular velocity around a specified axis.
    
    Parameters:
        rotation_period_days: Full rotation period [days]
        alpha: Tilt angle of rotation axis [radians]
        
    Default: 12-day rotation around polar axis (alpha=0)
    Williamson standard: alpha = π/2 - 0.05 (nearly equatorial)
    
    Purpose: Test advection, test case 1 standard."""


def deformational_flow(geometry: CubedSphereGeometry,
                       planet: PlanetParams,
                       period_days: float = 5.0,
                       time: float = 0.0,
                       dtype=None) -> tuple:
    """
    Time-dependent deformational flow (Lauritzen/Nair test case).
    
    A more challenging test with time-dependent deformation that
    stretches and folds the tracer field.
    
    Args:
        geometry: CubedSphereGeometry instance
        planet: PlanetParams
        period_days: Flow reversal period [days]
        time: Current time [s]
        dtype: Output dtype
        
    Returns:
        Vx, Vy, Vz: Cartesian velocity components [m/s]
        
    Reference: Lauritzen et al. (2012), GMD
    """
    # TODO: Implement deformational flow test case
    raise NotImplementedError("Deformational flow not yet implemented")

