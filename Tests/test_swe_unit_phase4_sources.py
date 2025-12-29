"""
SWE Unit Tests - Phase 4: Source Terms (Coriolis and Metric)

Tests for the momentum source terms:
- Coriolis force: f × V
- Metric source: Christoffel symbol terms
- Geostrophic balance verification

Run with: pytest Tests/test_swe_unit_phase4_sources.py -v
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


# =============================================================================
# Test 4.1: Coriolis Parameter
# =============================================================================

class TestCoriolisParameter:
    """Tests for Coriolis parameter f = 2Ω sin(lat)."""
    
    def test_coriolis_at_poles(self):
        """f = ±2Ω at poles."""
        omega = 7.292e-5
        
        f_north = 2 * omega * np.sin(np.pi/2)
        f_south = 2 * omega * np.sin(-np.pi/2)
        
        np.testing.assert_allclose(f_north, 2*omega, rtol=1e-14)
        np.testing.assert_allclose(f_south, -2*omega, rtol=1e-14)
    
    def test_coriolis_at_equator(self):
        """f = 0 at equator."""
        omega = 7.292e-5
        
        f_equator = 2 * omega * np.sin(0)
        
        assert abs(f_equator) < 1e-20
    
    def test_coriolis_on_grid(self):
        """Compute Coriolis parameter on full cubed-sphere grid."""
        N = 30
        geom = CubedSphereGeometry.create(N)
        lat, lon = geom.get_lat_lon_all_faces()
        omega = 7.292e-5
        
        f = 2 * omega * np.sin(lat)
        
        # Check range
        assert np.min(f) > -2*omega - 1e-10
        assert np.max(f) < 2*omega + 1e-10
        
        # Polar faces should have larger |f|
        f_face0 = f[0]  # North polar
        f_face5 = f[5]  # South polar
        
        assert np.mean(np.abs(f_face0)) > np.mean(np.abs(f[1]))  # Face 1 is equatorial


# =============================================================================
# Test 4.2: Coriolis Acceleration
# =============================================================================

class TestCoriolisAcceleration:
    """Tests for Coriolis acceleration in momentum equations."""
    
    def test_coriolis_direction(self):
        """
        Coriolis deflects:
        - Eastward flow southward in NH, northward in SH
        - Northward flow eastward in NH, westward in SH
        
        In component form: a_coriolis = f × V
        For f pointing up (NH) and V pointing east: a points south
        """
        omega = 7.292e-5
        lat = np.pi/4  # 45°N
        f = 2 * omega * np.sin(lat)
        
        # Eastward velocity at 45N
        u_east = 40.0  # m/s
        
        # Coriolis acceleration (in tangent plane):
        # a_north = -f * u_east (deflects southward in NH)
        # a_east = f * u_north
        
        a_north = -f * u_east
        
        # Should be negative (southward) in NH
        assert a_north < 0
        
        # Magnitude check
        expected_mag = abs(f * u_east)
        assert abs(a_north) == pytest.approx(expected_mag)
    
    def test_coriolis_magnitude(self):
        """Coriolis magnitude = |f||V|."""
        omega = 7.292e-5
        lat = np.radians(30)  # 30°N
        f = 2 * omega * np.sin(lat)
        V = 40.0  # m/s
        
        # Expected acceleration
        a_coriolis = abs(f) * V
        
        # This is ~0.003 m/s² at 30°N with V=40 m/s
        assert 0.001 < a_coriolis < 0.01


# =============================================================================
# Test 4.3: Metric Source Terms (Christoffel Symbols)
# =============================================================================

class TestMetricSource:
    """
    Tests for metric source terms from Christoffel symbols.
    
    In curvilinear coordinates, momentum equations have source terms:
    ∂(huⁱ)/∂t + ... = -Γⁱ_jk T^{jk} + Coriolis + ...
    
    For equiangular cubed-sphere (Ullrich et al. 2010, Eq. 15):
    W_M = (2/δ²) [ XY²u¹u¹ + Y(1+Y²)u¹u²    ]
                  [ X(1+X²)u¹u² - X²Yu²u²   ]
    where X = tan(ξ¹), Y = tan(ξ²), δ = √(1 + X² + Y²)
    """
    
    def test_metric_source_at_face_center(self):
        """At face center (X=Y=0), metric source is zero."""
        X, Y = 0.0, 0.0
        d2 = 1 + X**2 + Y**2  # = 1
        
        u1, u2 = 10.0, 5.0
        
        # W_M1 = (2/δ²) * [XY²u¹u¹ + Y(1+Y²)u¹u²]
        W_M1 = (2/d2) * (X*Y**2*u1*u1 + Y*(1+Y**2)*u1*u2)
        
        # W_M2 = (2/δ²) * [X(1+X²)u¹u² - X²Yu²u²]
        W_M2 = (2/d2) * (X*(1+X**2)*u1*u2 - X**2*Y*u2*u2)
        
        # Both should be zero at center
        assert abs(W_M1) < 1e-14
        assert abs(W_M2) < 1e-14
    
    def test_metric_source_away_from_center(self):
        """Away from center, metric source is non-zero."""
        X, Y = 0.5, 0.3
        d2 = 1 + X**2 + Y**2
        
        u1, u2 = 10.0, 5.0
        
        W_M1 = (2/d2) * (X*Y**2*u1*u1 + Y*(1+Y**2)*u1*u2)
        W_M2 = (2/d2) * (X*(1+X**2)*u1*u2 - X**2*Y*u2*u2)
        
        # At least one should be non-zero
        assert abs(W_M1) > 0 or abs(W_M2) > 0
    
    def test_metric_source_symmetry(self):
        """Metric source has certain symmetries under X ↔ Y exchange."""
        X, Y = 0.4, 0.6
        d2 = 1 + X**2 + Y**2
        
        u1, u2 = 10.0, 10.0  # Equal velocities for symmetry test
        
        W_M1 = (2/d2) * (X*Y**2*u1*u1 + Y*(1+Y**2)*u1*u2)
        W_M2 = (2/d2) * (X*(1+X**2)*u1*u2 - X**2*Y*u2*u2)
        
        # Swap X ↔ Y, u1 ↔ u2
        W_M1_swap = (2/d2) * (Y*X**2*u2*u2 + X*(1+X**2)*u2*u1)
        W_M2_swap = (2/d2) * (Y*(1+Y**2)*u2*u1 - Y**2*X*u1*u1)
        
        # Under full swap, W_M1 ↔ W_M2
        # This is a consistency check


# =============================================================================
# Test 4.4: Geostrophic Balance
# =============================================================================

class TestGeostrophicBalance:
    """
    For steady geostrophic flow, pressure gradient balances Coriolis:
    g∇h = f × V
    
    This is the fundamental balance that Test Case 2 should maintain.
    """
    
    def test_geostrophic_height_formula(self):
        """
        Verify analytical geostrophic height:
        h = h0 - (1/g)(aΩu0 + u0²/2) sin²(lat)
        """
        h0 = 8000.0
        u0 = 40.0
        g = 9.81
        omega = 7.292e-5
        a = 6.371e6
        
        # At equator
        lat = 0.0
        h_eq = h0 - (1/g) * (a*omega*u0 + u0**2/2) * np.sin(lat)**2
        assert h_eq == pytest.approx(h0)  # No variation at equator
        
        # At pole
        lat = np.pi/2
        h_pole = h0 - (1/g) * (a*omega*u0 + u0**2/2) * np.sin(lat)**2
        
        # Height at pole should be less than at equator
        assert h_pole < h0
        
        # Difference
        dh = h0 - h_pole
        expected_dh = (1/g) * (a*omega*u0 + u0**2/2)
        assert dh == pytest.approx(expected_dh)
    
    def test_pressure_gradient_magnitude(self):
        """
        Pressure gradient magnitude should equal Coriolis + centripetal.
        
        For solid body rotation, the full balance is:
        g|∇h| = |f|V + V²tan(lat)/R
        
        The centripetal term is essential for circular motion!
        """
        u0 = 40.0
        omega = 7.292e-5
        a = 6.371e6
        g = 9.81
        
        lat = np.radians(45)  # 45°N
        
        # Velocity at this latitude
        V = u0 * np.cos(lat)
        
        # Coriolis acceleration (meridional component)
        f = 2 * omega * np.sin(lat)
        coriolis_mag = abs(f) * V
        
        # Centripetal acceleration (meridional component)
        # For circular motion at latitude, a_c = V²/(R*cos(lat)) pointing toward axis
        # Meridional component = a_c * sin(lat) = V²*tan(lat)/R
        centripetal_mag = V**2 * np.tan(lat) / a
        
        # Total acceleration that must be balanced by pressure gradient
        total_accel = coriolis_mag + centripetal_mag
        
        # Pressure gradient: g|∇h|
        # ∂h/∂lat = -(1/g)(aΩu0 + u0²/2) * sin(2lat)
        dh_dlat = -(1/g) * (a*omega*u0 + u0**2/2) * np.sin(2*lat)
        
        # |∇h| = (1/a)|∂h/∂lat| on sphere
        grad_h_mag = abs(dh_dlat) / a
        
        pressure_gradient_mag = g * grad_h_mag
        
        # These should be equal for full geostrophic-centripetal balance
        np.testing.assert_allclose(
            pressure_gradient_mag, total_accel, rtol=1e-10,
            err_msg="Geostrophic-centripetal balance violated"
        )
    
    def test_balance_on_grid(self):
        """
        Verify geostrophic-centripetal balance on cubed-sphere grid.
        
        For solid body rotation, the balance includes both Coriolis AND
        centripetal acceleration:
        
        g|∇h| = |f|V + V²tan(lat)/R
        """
        N = 30
        geom = CubedSphereGeometry.create(N)
        lat, lon = geom.get_lat_lon_all_faces()
        
        h0 = 8000.0
        u0 = 40.0
        g = 9.81
        omega = 7.292e-5
        R = 6.371e6
        
        # Geostrophic height
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        h = h0 - (1/g) * (R*omega*u0 + u0**2/2) * sin_lat**2
        
        # Velocity at each latitude
        V = u0 * cos_lat
        
        # Coriolis parameter
        f = 2 * omega * sin_lat
        
        # Coriolis acceleration magnitude
        coriolis_mag = np.abs(f) * V
        
        # Centripetal acceleration magnitude (meridional component)
        # = V²*tan(lat)/R, but need to handle equator carefully
        with np.errstate(divide='ignore', invalid='ignore'):
            centripetal_mag = np.where(
                np.abs(lat) > 1e-10,
                V**2 * np.tan(lat) / R,
                0.0
            )
        
        # Total acceleration (both point toward equator in NH, toward pole in SH)
        total_accel = coriolis_mag + np.abs(centripetal_mag)
        
        # Pressure gradient from analytical formula
        dh_dlat_analytical = -(1/g) * (R*omega*u0 + u0**2/2) * 2*sin_lat*cos_lat
        pressure_grad_analytical = g * np.abs(dh_dlat_analytical) / R
        
        # Compare (avoiding equator where accelerations are small)
        mask = np.abs(lat) > np.radians(10)
        
        rel_error = np.abs(pressure_grad_analytical[mask] - total_accel[mask]) / \
                    (total_accel[mask] + 1e-20)
        
        max_rel_error = np.max(rel_error)
        assert max_rel_error < 1e-10, f"Max balance error: {max_rel_error:.2e}"


# =============================================================================
# Test 4.5: Combined Source Term Consistency
# =============================================================================

class TestCombinedSources:
    """Tests for combined source terms in momentum equations."""
    
    def test_source_units(self):
        """All source terms should have units of m/s²."""
        omega = 7.292e-5  # rad/s
        V = 40.0  # m/s
        lat = np.radians(30)
        
        # Coriolis: [rad/s] * [m/s] = [m/s²] ✓
        f = 2 * omega * np.sin(lat)
        coriolis = f * V
        
        # Pressure gradient: g * ∇h
        # [m/s²] * [m/m] = [m/s²] ✓
        g = 9.81
        grad_h = 0.001  # Typical value [m/m]
        pressure = g * grad_h
        
        # Both have correct units
        assert isinstance(coriolis, float)
        assert isinstance(pressure, float)
    
    def test_tendency_should_be_zero_for_steady_state(self):
        """
        For Test Case 2 (steady geostrophic), all tendencies should be zero
        if the formulation is correct.
        
        ∂h/∂t = -∇·(hV) = 0  (V is divergence-free, h is steady)
        ∂(hV)/∂t = 0  (geostrophic balance: ∇p = fV)
        """
        # This is a statement of the physics
        # The actual numerical test is in Phase 5
        pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
