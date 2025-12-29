"""
SWE Unit Tests - Phase 2: Velocity Transformations

These tests validate velocity coordinate transformations for the SWE solver.

KEY INSIGHT about the coordinate transform bug:
- Spherical velocity (u_lon, u_lat) is in PHYSICAL units: m/s
- Contravariant velocity (u¹, u²) is in ANGULAR units: ~rad/s when R is used
- The Jacobian with R=6.371e6 gives: u_contravariant ≈ V_cartesian / R

The SWE solver BUG is treating u_lon (m/s) as if it were u¹ (rad/s).
This creates a ~6 million factor error in momentum fluxes!

Run with: pytest test_swe_unit_phase2_velocity.py -v -s
"""

import sys
from pathlib import Path
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from Solvers.geometry import CubedSphereGeometry
from Solvers.geometry.cubesphere import xi_to_xyz_face
from Solvers.physics import PlanetParams, EARTH
from Solvers.fv_plr_cubesphere_adv import (
    cartesian_to_contravariant,
    contravariant_to_cartesian,
    compute_jacobian_face,
    solid_body_rotation_velocity
)


# =============================================================================
# Test 2.1: Spherical to Cartesian Velocity Transform
# =============================================================================

class TestSphericalToCartesian:
    """Verify spherical (u_lon, u_lat) → Cartesian (Vx, Vy, Vz) transform."""
    
    @staticmethod
    def spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat):
        """
        Transform spherical velocity components to Cartesian.
        
        u_lon: eastward velocity [m/s]
        u_lat: northward velocity [m/s]
        lon, lat: position [radians]
        
        Returns: Vx, Vy, Vz [m/s]
        """
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        
        Vx = -u_lon * sin_lon - u_lat * sin_lat * cos_lon
        Vy = u_lon * cos_lon - u_lat * sin_lat * sin_lon
        Vz = u_lat * cos_lat
        
        return Vx, Vy, Vz
    
    def test_zonal_flow_at_equator(self):
        """Pure eastward flow at equator, lon=0: u_lon → Vy."""
        u_lon, u_lat = 40.0, 0.0
        lon, lat = 0.0, 0.0  # On equator at prime meridian
        
        Vx, Vy, Vz = self.spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat)
        
        # At (lon=0, lat=0): position is (1, 0, 0)
        # Eastward is +Y direction
        np.testing.assert_allclose([Vx, Vy, Vz], [0, 40, 0], atol=1e-12)
    
    def test_zonal_flow_at_90E(self):
        """Pure eastward flow at lon=90°: u_lon → -Vx."""
        u_lon, u_lat = 40.0, 0.0
        lon, lat = np.pi/2, 0.0  # Equator at 90°E
        
        Vx, Vy, Vz = self.spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat)
        
        # At (lon=90°, lat=0): position is (0, 1, 0)
        # Eastward is -X direction
        np.testing.assert_allclose([Vx, Vy, Vz], [-40, 0, 0], atol=1e-12)
    
    def test_meridional_flow_at_equator(self):
        """Pure northward flow at equator: u_lat → Vz."""
        u_lon, u_lat = 0.0, 40.0
        lon, lat = 0.0, 0.0
        
        Vx, Vy, Vz = self.spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat)
        
        # At equator, northward is +Z
        np.testing.assert_allclose([Vx, Vy, Vz], [0, 0, 40], atol=1e-12)
    
    def test_solid_body_rotation_cartesian(self):
        """Solid body rotation V = u0 * cos(lat) eastward → V = ω × r."""
        u0 = 40.0
        
        # Test at lon=0, lat=0 (position (1,0,0))
        u_lon = u0 * np.cos(0)
        Vx, Vy, Vz = self.spherical_to_cartesian_velocity(u_lon, 0, 0, 0)
        np.testing.assert_allclose([Vx, Vy, Vz], [0, u0, 0], atol=1e-12)
        
        # Test at lon=90°, lat=0 (position (0,1,0))
        Vx, Vy, Vz = self.spherical_to_cartesian_velocity(u_lon, 0, np.pi/2, 0)
        np.testing.assert_allclose([Vx, Vy, Vz], [-u0, 0, 0], atol=1e-12)


# =============================================================================
# Test 2.2: Jacobian and Contravariant Transform
# =============================================================================

class TestCartesianToContravariant:
    """
    Verify Cartesian → contravariant transform used by advection solver.
    
    CRITICAL: The Jacobian includes R=6.371e6, so:
    - Cartesian velocity ~ 40 m/s
    - Contravariant velocity ~ 40 / 6.371e6 ~ 6e-6 rad/s
    """
    
    def test_jacobian_magnitude(self):
        """Jacobian elements should be O(R)."""
        xi1 = jnp.array([[0.0]])
        xi2 = jnp.array([[0.0]])
        R = 6.371e6
        
        for face in range(6):
            J = compute_jacobian_face(xi1, xi2, face, R)
            J_vals = [float(j.flatten()[0]) for j in J]
            
            # At least one element should be O(R)
            max_J = max(abs(j) for j in J_vals)
            assert max_J > 0.1 * R, f"Jacobian too small on face {face}"
            assert max_J < 10 * R, f"Jacobian too large on face {face}"
    
    def test_round_trip_face_center(self):
        """contravariant → Cartesian → contravariant is identity at face center."""
        xi1 = jnp.array([[0.0]])
        xi2 = jnp.array([[0.0]])
        
        for face in range(6):
            # Use REALISTIC contravariant velocity (angular, ~1e-6 rad/s)
            u1_orig = jnp.array([[1e-6]])
            u2_orig = jnp.array([[0.5e-6]])
            
            # To Cartesian
            Vx, Vy, Vz = contravariant_to_cartesian(u1_orig, u2_orig, xi1, xi2, face)
            
            # Back to contravariant
            u1_back, u2_back = cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face)
            
            # Extract scalar values properly from JAX arrays
            u1_back_val = float(u1_back.flatten()[0])
            u2_back_val = float(u2_back.flatten()[0])
            u1_orig_val = float(u1_orig.flatten()[0])
            u2_orig_val = float(u2_orig.flatten()[0])
            
            np.testing.assert_allclose(
                [u1_back_val, u2_back_val],
                [u1_orig_val, u2_orig_val],
                rtol=1e-10,
                err_msg=f"Round-trip failed on face {face}"
            )
    
    def test_round_trip_off_center(self):
        """Round-trip works away from face center."""
        xi1 = jnp.array([[0.3]])
        xi2 = jnp.array([[-0.2]])
        
        for face in range(6):
            u1_orig = jnp.array([[1.5e-6]])
            u2_orig = jnp.array([[2.3e-6]])
            
            Vx, Vy, Vz = contravariant_to_cartesian(u1_orig, u2_orig, xi1, xi2, face)
            u1_back, u2_back = cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face)
            
            u1_back_val = float(u1_back.flatten()[0])
            u2_back_val = float(u2_back.flatten()[0])
            u1_orig_val = float(u1_orig.flatten()[0])
            u2_orig_val = float(u2_orig.flatten()[0])
            
            np.testing.assert_allclose(
                [u1_back_val, u2_back_val],
                [u1_orig_val, u2_orig_val],
                rtol=1e-10,
                err_msg=f"Round-trip failed on face {face}"
            )
    
    def test_velocity_scaling_with_R(self):
        """
        Contravariant velocity ~ Cartesian velocity / R.
        
        This is the KEY relationship that the SWE bug ignores!
        """
        xi1 = jnp.array([[0.0]])
        xi2 = jnp.array([[0.0]])
        R = 6.371e6
        face = 4  # +X equatorial face
        
        # Physical Cartesian velocity
        u0 = 40.0  # m/s
        Vx, Vy, Vz = jnp.array([[0.0]]), jnp.array([[u0]]), jnp.array([[0.0]])
        
        # Convert to contravariant
        u1, u2 = cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face)
        
        u1_val = abs(float(u1.flatten()[0]))
        u2_val = abs(float(u2.flatten()[0]))
        max_u = max(u1_val, u2_val)
        
        # Contravariant should be O(u0/R) ~ 6e-6
        expected_scale = u0 / R
        
        print(f"\nVelocity scaling test:")
        print(f"  V_cartesian = {u0} m/s")
        print(f"  u_contravariant max = {max_u:.2e} rad/s")
        print(f"  Expected scale: ~{expected_scale:.2e}")
        
        # Should be within an order of magnitude
        assert max_u < 100 * expected_scale, f"Contravariant too large: {max_u}"
        assert max_u > 0.01 * expected_scale, f"Contravariant too small: {max_u}"


# =============================================================================
# Test 2.3: The BUG - SWE Uses Wrong Velocity Units
# =============================================================================

class TestSphericalVsContravariant:
    """
    THIS IS THE BUG: SWE treats u_lon (m/s) as if it were u¹ (rad/s).
    
    The SWE solver does: hu1 = h * u_lon  (in m²/s)
    But flux expects:     hu1 = h * u¹    (in m·rad/s)
    
    The mismatch is a factor of R ~ 6.371e6!
    """
    
    def test_unit_mismatch_magnitude(self):
        """
        The buggy initialization uses velocities that are ~R times too large.
        """
        R = 6.371e6
        u0 = 40.0  # Physical velocity in m/s
        
        # What buggy code uses (treating u_lon as u¹)
        u1_wrong = u0  # This is in m/s
        
        # What correct code should use
        u1_correct = u0 / R  # This is in rad/s (approximately)
        
        ratio = u1_wrong / u1_correct
        
        print(f"\n" + "="*60)
        print("BUG DEMONSTRATION: Velocity Unit Mismatch")
        print("="*60)
        print(f"Buggy u1 (=u_lon):     {u1_wrong:.2f} m/s")
        print(f"Correct u1 (=u_contra): {u1_correct:.2e} rad/s")
        print(f"Error ratio:            {ratio:.2e} (≈ R)")
        print("="*60)
        
        # The ratio should be approximately R
        assert 0.5 * R < ratio < 2 * R, f"Ratio {ratio} should be ~R"
    
    def test_momentum_flux_error(self):
        """
        The momentum flux h*u¹ is wrong by factor ~R due to velocity bug.
        """
        h = 8000.0  # meters
        u0 = 40.0  # m/s
        R = 6.371e6
        
        # Buggy flux (what SWE computes)
        hu1_wrong = h * u0  # h * u_lon (wrong units)
        
        # Correct flux
        u1_correct = u0 / R
        hu1_correct = h * u1_correct
        
        error_factor = hu1_wrong / hu1_correct
        
        print(f"\nMomentum flux error:")
        print(f"  Buggy hu1 = h * u_lon = {hu1_wrong:.2e}")
        print(f"  Correct hu1 = h * u¹  = {hu1_correct:.2e}")
        print(f"  Error factor: {error_factor:.2e}")
        
        # This explains the negative convergence rate!
        assert error_factor > 1e5, "Error should be huge"


# =============================================================================
# Test 2.4: Advection Solver Pattern (Working Reference)
# =============================================================================

class TestAdvectionPattern:
    """
    The advection solver stores velocity in Cartesian form (Vx, Vy, Vz)
    and converts to contravariant only at flux computation time.
    
    This is the correct pattern that SWE should follow.
    """
    
    def test_advection_velocity_is_cartesian(self):
        """Advection stores (Vx, Vy, Vz) in m/s, which is correct."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        XI1 = jnp.array(geom.XI1)
        XI2 = jnp.array(geom.XI2)
        u0 = 40.0
        
        # Use an equatorial face (not polar) for meaningful velocity
        face = 4  # +X face (equatorial)
        Vx, Vy, Vz = solid_body_rotation_velocity(XI1, XI2, face_id=face, u0=u0)
        
        # For solid body rotation V = u0 * (-Y, X, 0) / |r| on unit sphere
        # With R, it's V = u0 * (-Y, X, 0) in m/s
        X, Y, Z = xi_to_xyz_face(np.array(XI1), np.array(XI2), face)
        
        # Expected: Vx = -u0*Y, Vy = u0*X, Vz = 0
        Vx_expected = -u0 * Y
        Vy_expected = u0 * X
        
        np.testing.assert_allclose(np.array(Vx), Vx_expected, rtol=1e-10)
        np.testing.assert_allclose(np.array(Vy), Vy_expected, rtol=1e-10)
        np.testing.assert_allclose(np.array(Vz), np.zeros_like(X), atol=1e-10)
        
        # Verify magnitude is O(u0)
        V_mag = np.sqrt(np.array(Vx)**2 + np.array(Vy)**2 + np.array(Vz)**2)
        assert V_mag.max() < 2 * u0, "Velocity should be ~u0"
        assert V_mag.max() > 0.1 * u0, "Velocity should be significant"
    
    def test_advection_converts_at_flux_time(self):
        """At flux time, advection converts Cartesian → contravariant."""
        N = 30
        R = 6.371e6
        geom = CubedSphereGeometry.create(N)
        XI1 = jnp.array(geom.XI1)
        XI2 = jnp.array(geom.XI2)
        u0 = 40.0
        
        # Use EQUATORIAL face where velocity is significant
        face = 4  # +X face
        
        # Get Cartesian velocity (in m/s)
        Vx, Vy, Vz = solid_body_rotation_velocity(XI1, XI2, face, u0=u0)
        
        # At flux computation, convert to contravariant
        u1, u2 = cartesian_to_contravariant(Vx, Vy, Vz, XI1, XI2, face)
        
        # Check shapes
        assert u1.shape == (N, N)
        assert u2.shape == (N, N)
        
        # Contravariant velocities should be O(u0/R) ~ 6e-6 rad/s
        u1_max = float(np.max(np.abs(np.array(u1))))
        u2_max = float(np.max(np.abs(np.array(u2))))
        expected_scale = u0 / R
        
        print(f"\nAdvection flux-time conversion:")
        print(f"  max|u¹| = {u1_max:.2e}")
        print(f"  max|u²| = {u2_max:.2e}")
        print(f"  Expected scale: ~{expected_scale:.2e}")
        
        # At least one component should be significant (order of u0/R)
        max_u = max(u1_max, u2_max)
        assert max_u > 0.01 * expected_scale, \
            f"Contravariant velocity too small: {max_u}"
        assert max_u < 100 * expected_scale, \
            f"Contravariant velocity too large: {max_u}"


# =============================================================================
# Test 2.5: Correct Velocity Initialization for SWE
# =============================================================================

class TestCorrectVelocityInit:
    """
    Tests demonstrating the CORRECT way to initialize SWE velocities.
    
    The fix: Store Cartesian momentum (h*Vx, h*Vy, h*Vz) like advection,
    then convert to contravariant only at flux computation time.
    """
    
    def test_correct_init_stores_cartesian(self):
        """
        SWE should store Cartesian momentum, not spherical velocity.
        """
        N = 10
        geom = CubedSphereGeometry.create(N)
        lat, lon = geom.get_lat_lon_all_faces()
        
        h0 = 8000.0
        u0 = 40.0
        
        # Spherical velocity from analytical solution
        u_lon = u0 * np.cos(lat)  # m/s
        u_lat = np.zeros_like(lat)
        
        # Convert to Cartesian (THIS IS THE FIX)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        
        Vx = -u_lon * sin_lon - u_lat * sin_lat * cos_lon
        Vy = u_lon * cos_lon - u_lat * sin_lat * sin_lon
        Vz = u_lat * cos_lat
        
        # Cartesian momentum (what SWE SHOULD store)
        hVx = h0 * Vx
        hVy = h0 * Vy
        hVz = h0 * Vz
        
        # Verify shapes and magnitudes
        assert hVx.shape == (6, N, N)
        
        V_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        hV_mag = np.sqrt(hVx**2 + hVy**2 + hVz**2)
        
        print(f"\nCorrect initialization pattern:")
        print(f"  |V| max: {V_mag.max():.1f} m/s")
        print(f"  |hV| max: {hV_mag.max():.2e} m²/s")
        
        # V should be physical velocity (~u0)
        assert V_mag.max() < 2 * u0
        # hV should be h * V
        np.testing.assert_allclose(hV_mag, h0 * V_mag, rtol=1e-10)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
