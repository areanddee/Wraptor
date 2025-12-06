"""
Initial Conditions Package for JaxStream2 Solvers

Provides analytic initial conditions for various test cases.
All ICs take geometry (for coordinates) and planet params (for physics).

Usage:
    from Solvers.geometry import CubedSphereGeometry
    from Solvers.physics import EARTH
    from Solvers.initial_conditions import lima_flag, cosine_bell
    
    geometry = CubedSphereGeometry.create(N=60)
    
    # Diffusion test
    T = lima_flag(geometry, T_hot=600.0, T_background=1.0)
    
    # Advection test
    q = cosine_bell(geometry, EARTH, amplitude=1000.0)
"""

from .lima_flag import lima_flag
from .cosine_bell import cosine_bell
from .velocity_fields import solid_body_rotation

__all__ = ['lima_flag', 'cosine_bell', 'solid_body_rotation']

