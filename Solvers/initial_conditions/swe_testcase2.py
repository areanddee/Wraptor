"""
Initial conditions for shallow-water Test Case 2 (Williamson et al. 1992).

Implements the steady geostrophic zonal-flow solution on the rotating sphere.
"""

from typing import Tuple

import jax.numpy as jnp

from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams


def steady_geostrophic_flow(geometry: CubedSphereGeometry,
                            planet: PlanetParams,
                            h0: float = 8000.0,
                            u0: float = 40.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute depth and zonal velocity for Test Case 2.

    Args:
        geometry: Cubed-sphere geometry (unit sphere).
        planet: Planet parameters (radius, gravity, rotation).
        h0: Mean fluid depth [m].
        u0: Characteristic zonal speed at the equator [m/s].

    Returns:
        h: (6, N, N) fluid depth.
        u_lon: (6, N, N) zonal velocity (eastward) on the sphere.
        u_lat: (6, N, N) meridional velocity (northward); identically zero.
    """
    lat, lon = geometry.get_lat_lon_all_faces()
    a = planet.R_sphere
    g = planet.gravity
    omega = planet.omega

    # Steady analytic depth (Williamson et al. 1992, Test Case 2)
    # h = h0 - (1/g) * (a*Ω*u0 + u0^2/2) * sin^2(φ)
    sin_lat = jnp.sin(lat)
    h = h0 - (1.0 / g) * (a * omega * u0 + u0**2 / 2.0) * sin_lat**2

    # Solid body zonal flow (eastward) that depends on latitude
    u_lon = u0 * jnp.cos(lat)
    u_lat = jnp.zeros_like(u_lon)

    return h, u_lon, u_lat

