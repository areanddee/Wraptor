"""
Geometry Package for JaxStream2 Solvers

This package provides geometry primitives for different grid types.
Geometry is computed in float64 for accuracy, regardless of solver precision.

Currently supported:
- CubedSphereGeometry: 6-face equiangular gnomonic cubed-sphere

Design principles:
- Geometry is dimensionless (unit sphere, R=1)
- Physical scaling (planet radius, etc.) is applied by the solver/IC
- All arrays stored in float64
- Immutable after creation
"""

from .cubesphere import CubedSphereGeometry

__all__ = ['CubedSphereGeometry']

