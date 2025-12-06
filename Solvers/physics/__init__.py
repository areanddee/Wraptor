"""
Physics Package for JaxStream2 Solvers

Contains physical constants and planet parameters.
These are the SINGLE SOURCE OF TRUTH for physical values.

Usage:
    from Solvers.physics import PlanetParams
    
    planet = PlanetParams.from_config(config)
    dx_meters = geometry.dx * planet.R_sphere
"""

from .planet import PlanetParams, EARTH, MARS

__all__ = ['PlanetParams', 'EARTH', 'MARS']

