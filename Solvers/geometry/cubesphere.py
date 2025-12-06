"""
Cubed-Sphere Geometry

Equiangular gnomonic projection on 6 cube faces.
All computations on UNIT SPHERE (R=1). Physical scaling applied externally.

Coordinate system:
- ξ¹, ξ² ∈ [-π/4, π/4] on each face
- Face 0: +Z (north pole)
- Face 1: +Y
- Face 2: -X  
- Face 3: -Y
- Face 4: +X
- Face 5: -Z (south pole)

Reference: Ronchi, Iacono, Paolucci (1996)
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class CubedSphereGeometry:
    """
    Cubed-sphere geometry on unit sphere.
    
    All arrays are float64 for accuracy.
    Immutable after creation (frozen dataclass).
    
    Attributes:
        N: Grid points per face edge (interior only)
        n_faces: Number of faces (always 6 for cubed-sphere)
        dx: Angular grid spacing [radians]
        xi1: (N,) coordinate array in first direction
        xi2: (N,) coordinate array in second direction
        XI1: (N, N) meshgrid of xi1
        XI2: (N, N) meshgrid of xi2
        sqrtG: (6, N, N) metric determinant √G on all faces
        
    Note: Physical quantities like dx_meters require R_sphere from PlanetParams.
    """
    N: int
    n_faces: int
    dx: float
    xi1: np.ndarray
    xi2: np.ndarray
    XI1: np.ndarray
    XI2: np.ndarray
    sqrtG: np.ndarray
    
    @classmethod
    def create(cls, N: int) -> 'CubedSphereGeometry':
        """
        Factory method to create CubedSphereGeometry.
        
        Args:
            N: Grid points per face edge
            
        Returns:
            CubedSphereGeometry instance with all arrays in float64
        """
        n_faces = 6
        dx = np.pi / (2 * N)
        
        # 1D coordinate arrays (cell centers)
        xi1 = np.linspace(-np.pi/4 + dx/2, np.pi/4 - dx/2, N, dtype=np.float64)
        xi2 = np.linspace(-np.pi/4 + dx/2, np.pi/4 - dx/2, N, dtype=np.float64)
        
        # 2D meshgrid
        XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
        
        # Metric determinant on all faces
        sqrtG = np.zeros((n_faces, N, N), dtype=np.float64)
        for face in range(n_faces):
            sqrtG[face] = compute_metric(XI1, XI2)
        
        return cls(
            N=N,
            n_faces=n_faces,
            dx=dx,
            xi1=xi1,
            xi2=xi2,
            XI1=XI1,
            XI2=XI2,
            sqrtG=sqrtG
        )
    
    def xi_to_xyz(self, xi1: np.ndarray, xi2: np.ndarray, face: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map equiangular coordinates to Cartesian on unit sphere.
        
        Args:
            xi1, xi2: Coordinate arrays
            face: Face ID (0-5)
            
        Returns:
            X, Y, Z: Cartesian coordinates on unit sphere
        """
        return xi_to_xyz_face(xi1, xi2, face)
    
    def get_xyz_all_faces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Cartesian coordinates for all grid points on all faces.
        
        Returns:
            X, Y, Z: (6, N, N) arrays of Cartesian coordinates
        """
        X_all = np.zeros((self.n_faces, self.N, self.N), dtype=np.float64)
        Y_all = np.zeros((self.n_faces, self.N, self.N), dtype=np.float64)
        Z_all = np.zeros((self.n_faces, self.N, self.N), dtype=np.float64)
        
        for face in range(self.n_faces):
            X, Y, Z = self.xi_to_xyz(self.XI1, self.XI2, face)
            X_all[face] = X
            Y_all[face] = Y
            Z_all[face] = Z
        
        return X_all, Y_all, Z_all
    
    def get_lat_lon_all_faces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latitude/longitude for all grid points on all faces.
        
        Returns:
            lat, lon: (6, N, N) arrays in radians
        """
        X, Y, Z = self.get_xyz_all_faces()
        lat = np.arcsin(Z)  # Z on unit sphere = sin(lat)
        lon = np.arctan2(Y, X)
        return lat, lon
    
    def to_jax(self) -> 'CubedSphereGeometry':
        """
        Convert numpy arrays to JAX arrays.
        
        Returns:
            New CubedSphereGeometry with JAX arrays
        """
        return CubedSphereGeometry(
            N=self.N,
            n_faces=self.n_faces,
            dx=self.dx,
            xi1=jnp.array(self.xi1),
            xi2=jnp.array(self.xi2),
            XI1=jnp.array(self.XI1),
            XI2=jnp.array(self.XI2),
            sqrtG=jnp.array(self.sqrtG)
        )


# ============================================================================
# PURE FUNCTIONS (used by class and standalone)
# ============================================================================

def xi_to_xyz_face(xi1: np.ndarray, xi2: np.ndarray, face: int,
                   R: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map equiangular (ξ¹, ξ²) to Cartesian (X, Y, Z).
    
    Args:
        xi1, xi2: Equiangular coordinates in [-π/4, π/4]
        face: Face ID (0=+Z, 1=+Y, 2=-X, 3=-Y, 4=+X, 5=-Z)
        R: Sphere radius (default 1.0 for unit sphere)
        
    Returns:
        X, Y, Z: Cartesian coordinates
    """
    # Use numpy for geometry (always f64)
    tan_xi1 = np.tan(xi1)
    tan_xi2 = np.tan(xi2)
    delta = np.sqrt(1.0 + tan_xi1**2 + tan_xi2**2)
    
    if face == 0:  # +Z (north pole)
        X = R * tan_xi1 / delta
        Y = R * tan_xi2 / delta
        Z = R / delta
    elif face == 1:  # +Y
        X = -R * tan_xi1 / delta
        Y = R / delta
        Z = R * tan_xi2 / delta
    elif face == 2:  # -X
        X = -R / delta
        Y = -R * tan_xi1 / delta
        Z = R * tan_xi2 / delta
    elif face == 3:  # -Y
        X = R * tan_xi1 / delta
        Y = -R / delta
        Z = R * tan_xi2 / delta
    elif face == 4:  # +X
        X = R / delta
        Y = R * tan_xi1 / delta
        Z = R * tan_xi2 / delta
    elif face == 5:  # -Z (south pole)
        X = -R * tan_xi1 / delta
        Y = R * tan_xi2 / delta
        Z = -R / delta
    else:
        raise ValueError(f"Invalid face ID: {face}. Must be 0-5.")
    
    return X, Y, Z


def compute_metric(xi1: np.ndarray, xi2: np.ndarray, R: float = 1.0) -> np.ndarray:
    """
    Compute metric determinant √G for equiangular gnomonic projection.
    
    The metric is the same on all faces due to symmetry.
    
    √G = R² / (δ³ cos²ξ¹ cos²ξ²)
    
    where δ = √(1 + tan²ξ¹ + tan²ξ²)
    
    Args:
        xi1, xi2: Equiangular coordinates
        R: Sphere radius (default 1.0)
        
    Returns:
        sqrtG: Metric determinant
    """
    tan_xi1 = np.tan(xi1)
    tan_xi2 = np.tan(xi2)
    cos_xi1 = np.cos(xi1)
    cos_xi2 = np.cos(xi2)
    
    delta = np.sqrt(1.0 + tan_xi1**2 + tan_xi2**2)
    sqrtG = R**2 / (delta**3 * cos_xi1**2 * cos_xi2**2)
    
    return sqrtG


def xyz_to_latlon(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to lat/lon.
    
    Assumes unit sphere (or any sphere - lat/lon are angles).
    
    Args:
        X, Y, Z: Cartesian coordinates
        
    Returns:
        lat, lon: Latitude and longitude in radians
    """
    r = np.sqrt(X**2 + Y**2 + Z**2)
    lat = np.arcsin(Z / r)
    lon = np.arctan2(Y, X)
    return lat, lon


# ============================================================================
# JAX-COMPATIBLE VERSIONS (for use inside JIT-compiled functions)
# ============================================================================

def xi_to_xyz_face_jax(xi1, xi2, face: int, R: float = 1.0):
    """
    JAX version of xi_to_xyz_face for use in JIT-compiled code.
    
    Note: face must be a static integer (traced at compile time).
    """
    tan_xi1 = jnp.tan(xi1)
    tan_xi2 = jnp.tan(xi2)
    delta = jnp.sqrt(1.0 + tan_xi1**2 + tan_xi2**2)
    
    if face == 0:
        X, Y, Z = R * tan_xi1 / delta, R * tan_xi2 / delta, R / delta
    elif face == 1:
        X, Y, Z = -R * tan_xi1 / delta, R / delta, R * tan_xi2 / delta
    elif face == 2:
        X, Y, Z = -R / delta, -R * tan_xi1 / delta, R * tan_xi2 / delta
    elif face == 3:
        X, Y, Z = R * tan_xi1 / delta, -R / delta, R * tan_xi2 / delta
    elif face == 4:
        X, Y, Z = R / delta, R * tan_xi1 / delta, R * tan_xi2 / delta
    elif face == 5:
        X, Y, Z = -R * tan_xi1 / delta, R * tan_xi2 / delta, -R / delta
    else:
        raise ValueError(f"Invalid face: {face}")
    
    return X, Y, Z


def compute_metric_jax(xi1, xi2, R: float = 1.0):
    """JAX version of compute_metric for use in JIT-compiled code."""
    tan_xi1 = jnp.tan(xi1)
    tan_xi2 = jnp.tan(xi2)
    cos_xi1 = jnp.cos(xi1)
    cos_xi2 = jnp.cos(xi2)
    
    delta = jnp.sqrt(1.0 + tan_xi1**2 + tan_xi2**2)
    sqrtG = R**2 / (delta**3 * cos_xi1**2 * cos_xi2**2)
    
    return sqrtG

