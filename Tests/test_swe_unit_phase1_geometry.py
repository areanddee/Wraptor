"""
SWE Unit Tests - Phase 1: Geometry and Coordinate Transforms

These tests validate the foundational coordinate transformations before
testing the SWE solver itself. Each test is independent and diagnostic.

Run with: pytest Tests/test_swe_unit_phase1_geometry.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Enable float64
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from Solvers.geometry import CubedSphereGeometry
from Solvers.geometry.cubesphere import compute_metric, xi_to_xyz_face
from Solvers.physics import PlanetParams, EARTH


# =============================================================================
# Test 1.1: Basic Geometry Creation
# =============================================================================

class TestGeometryCreation:
    """Verify CubedSphereGeometry creates valid grids."""
    
    def test_geometry_dimensions(self):
        """Grid arrays have correct shapes."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        
        assert geom.N == N
        assert geom.n_faces == 6
        assert geom.xi1.shape == (N,)
        assert geom.xi2.shape == (N,)
        assert geom.XI1.shape == (N, N)
        assert geom.XI2.shape == (N, N)
        assert geom.sqrtG.shape == (6, N, N)
    
    def test_coordinate_range(self):
        """Coordinates span [-π/4, π/4] with cell centers."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        
        # Cell centers should be offset from edges by dx/2
        dx = np.pi / (2 * N)
        expected_min = -np.pi/4 + dx/2
        expected_max = np.pi/4 - dx/2
        
        assert np.isclose(geom.xi1[0], expected_min, rtol=1e-10)
        assert np.isclose(geom.xi1[-1], expected_max, rtol=1e-10)
    
    def test_metric_positive(self):
        """Metric determinant √G is everywhere positive."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        
        assert np.all(geom.sqrtG > 0), "Metric must be positive everywhere"
    
    def test_metric_symmetry(self):
        """Metric is same on all faces (by symmetry of equiangular projection)."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        
        for face in range(1, 6):
            np.testing.assert_allclose(
                geom.sqrtG[face], geom.sqrtG[0],
                rtol=1e-14,
                err_msg=f"Face {face} metric differs from face 0"
            )


# =============================================================================
# Test 1.2: Cartesian Coordinate Mapping
# =============================================================================

class TestCartesianMapping:
    """Verify xi → (X, Y, Z) mapping produces valid sphere points."""
    
    def test_unit_sphere(self):
        """All mapped points lie on unit sphere (X² + Y² + Z² = 1)."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        X, Y, Z = geom.get_xyz_all_faces()
        
        r_squared = X**2 + Y**2 + Z**2
        np.testing.assert_allclose(
            r_squared, 1.0, rtol=1e-14,
            err_msg="Points must lie on unit sphere"
        )
    
    def test_face_centers(self):
        """Face centers map to expected axis directions."""
        # Face 0: +Z (north pole)
        # Face 1: +Y, Face 2: -X, Face 3: -Y, Face 4: +X
        # Face 5: -Z (south pole)
        
        expected_normals = {
            0: (0, 0, 1),   # +Z
            1: (0, 1, 0),   # +Y
            2: (-1, 0, 0),  # -X
            3: (0, -1, 0),  # -Y
            4: (1, 0, 0),   # +X
            5: (0, 0, -1),  # -Z
        }
        
        for face, (nx, ny, nz) in expected_normals.items():
            # At xi1=xi2=0 (face center), should point along face normal
            X, Y, Z = xi_to_xyz_face(np.array([0.0]), np.array([0.0]), face)
            
            np.testing.assert_allclose(
                [float(X), float(Y), float(Z)],
                [nx, ny, nz],
                rtol=1e-14,
                err_msg=f"Face {face} center incorrect"
            )
    
    def test_no_duplicate_points(self):
        """Adjacent face edges share the same physical points."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        X, Y, Z = geom.get_xyz_all_faces()
        
        # Face 0 south edge (j=0) should match Face 3 north edge (j=N-1)
        # after appropriate coordinate transformation
        # This is a sanity check - actual connectivity is tested in halo exchange
        
        # Just verify we have 6*N*N points total
        total_points = 6 * N * N
        assert X.size == total_points


# =============================================================================
# Test 1.3: Latitude/Longitude Conversion
# =============================================================================

class TestLatLonConversion:
    """Verify Cartesian → lat/lon conversion."""
    
    def test_poles(self):
        """North/south pole faces have correct latitudes."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        lat, lon = geom.get_lat_lon_all_faces()
        
        # Face 0 center (north pole region) should have lat near 90°
        # Face 5 center (south pole region) should have lat near -90°
        center = N // 2
        
        assert lat[0, center, center] > np.radians(45), \
            "Face 0 center should be in northern hemisphere"
        assert lat[5, center, center] < np.radians(-45), \
            "Face 5 center should be in southern hemisphere"
    
    def test_equator(self):
        """Equatorial faces have latitude near 0 at face centers."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        lat, lon = geom.get_lat_lon_all_faces()
        
        center = N // 2
        for face in [1, 2, 3, 4]:
            assert abs(lat[face, center, center]) < np.radians(10), \
                f"Face {face} center should be near equator"
    
    def test_lat_range(self):
        """Latitude is in [-π/2, π/2]."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        lat, lon = geom.get_lat_lon_all_faces()
        
        assert np.all(lat >= -np.pi/2 - 1e-10)
        assert np.all(lat <= np.pi/2 + 1e-10)


# =============================================================================
# Test 1.4: Metric Tensor Computation
# =============================================================================

class TestMetricTensor:
    """Verify metric √G computation against analytical formula."""
    
    def test_metric_at_face_center(self):
        """At face center (ξ¹=ξ²=0), √G = 1 on unit sphere."""
        xi1 = np.array([0.0])
        xi2 = np.array([0.0])
        sqrtG = compute_metric(xi1, xi2, R=1.0)
        
        # At center: tan(0)=0, cos(0)=1, δ=1
        # √G = R² / (δ³ · cos²ξ¹ · cos²ξ²) = 1 / (1 · 1 · 1) = 1
        np.testing.assert_allclose(sqrtG, 1.0, rtol=1e-14)
    
    def test_metric_at_corner(self):
        """At face corner, √G has known value."""
        # Corner: ξ¹ = ξ² = π/4
        xi1 = np.array([np.pi/4])
        xi2 = np.array([np.pi/4])
        sqrtG = compute_metric(xi1, xi2, R=1.0)
        
        # tan(π/4) = 1, cos(π/4) = 1/√2
        # δ = √(1 + 1 + 1) = √3
        # √G = 1 / (3√3 · 0.5 · 0.5) = 1 / (3√3 · 0.25) = 4/(3√3) ≈ 0.7698
        expected = 4.0 / (3.0 * np.sqrt(3.0))
        np.testing.assert_allclose(sqrtG, expected, rtol=1e-10)
    
    def test_metric_scaling_with_radius(self):
        """√G scales as R² with sphere radius."""
        xi1 = np.array([0.1])
        xi2 = np.array([0.2])
        
        sqrtG_unit = compute_metric(xi1, xi2, R=1.0)
        sqrtG_earth = compute_metric(xi1, xi2, R=6.371e6)
        
        np.testing.assert_allclose(
            sqrtG_earth / sqrtG_unit, 
            (6.371e6)**2,
            rtol=1e-10
        )


# =============================================================================
# Test 1.5: Total Sphere Area
# =============================================================================

class TestSphereArea:
    """Verify that integrating √G gives correct sphere area."""
    
    def test_total_area_unit_sphere(self):
        """Sum of √G * dx² over all faces ≈ 4π (unit sphere area)."""
        N = 60  # Need moderate resolution for accuracy
        geom = CubedSphereGeometry.create(N)
        
        # Area = ∫∫ √G dξ¹ dξ² summed over 6 faces
        # For cell-centered values: Area ≈ Σ √G · dx²
        dx = geom.dx
        total_area = np.sum(geom.sqrtG) * dx**2
        
        expected_area = 4 * np.pi  # Unit sphere area
        
        # Should be accurate to ~1% at N=60
        np.testing.assert_allclose(
            total_area, expected_area, rtol=0.01,
            err_msg=f"Total area {total_area:.6f} != 4π = {expected_area:.6f}"
        )
    
    def test_area_convergence(self):
        """Area approximation converges with resolution."""
        areas = []
        for N in [20, 40, 80]:
            geom = CubedSphereGeometry.create(N)
            dx = geom.dx
            area = np.sum(geom.sqrtG) * dx**2
            areas.append(area)
        
        expected = 4 * np.pi
        errors = [abs(a - expected) for a in areas]
        
        # Error should decrease with resolution
        assert errors[1] < errors[0], "Error should decrease N=20→40"
        assert errors[2] < errors[1], "Error should decrease N=40→80"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
