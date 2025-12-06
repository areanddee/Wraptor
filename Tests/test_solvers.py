"""
Pytest Test Suite for JaxStream2 Solvers

Run with:
    pytest Tests/ -v                    # Full output
    pytest Tests/ -v --quick-only       # Quick tests only
    pytest Tests/ -v -k diffusion       # Only diffusion tests
    pytest Tests/ -v -k advection       # Only advection tests
    pytest Tests/ --tb=short            # Short tracebacks on failure

Dashboard view:
    pytest Tests/ --tb=no -q            # Minimal output, just pass/fail
"""

import pytest
import numpy as np
from pathlib import Path


# =============================================================================
# FRAMEWORK INTERFACE TESTS
# =============================================================================

class TestFrameworkInterface:
    """Test that solvers implement the Framework interface correctly."""
    
    @pytest.mark.quick
    def test_diffusion_interface(self, diffusion_solver_quick):
        """Diffusion solver implements all required Framework methods."""
        from Framework.solver_interface import validate_solver_interface
        validate_solver_interface(diffusion_solver_quick)
    
    @pytest.mark.quick
    def test_advection_interface(self, advection_solver_quick):
        """Advection solver implements all required Framework methods."""
        from Framework.solver_interface import validate_solver_interface
        validate_solver_interface(advection_solver_quick)
    
    @pytest.mark.quick
    def test_diffusion_output_groups(self, diffusion_solver_quick):
        """Diffusion solver defines expected output groups."""
        outputs = diffusion_solver_quick.get_available_outputs()
        assert 'state' in outputs
        assert 'diagnostics' in outputs
        assert 'T' in outputs['state']
    
    @pytest.mark.quick
    def test_advection_output_groups(self, advection_solver_quick):
        """Advection solver defines expected output groups."""
        outputs = advection_solver_quick.get_available_outputs()
        assert 'state' in outputs
        assert 'diagnostics' in outputs
        assert 'q' in outputs['state']


# =============================================================================
# RESTART TESTS
# =============================================================================

class TestRestartExactness:
    """Test that restart produces identical results to continuous run."""
    
    @pytest.mark.quick
    def test_diffusion_restart(self, diffusion_solver_quick):
        """Diffusion: stop + restart = continuous run."""
        solver = diffusion_solver_quick
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        
        dt = 1800.0
        total_steps = 20
        restart_step = 10
        
        # Run 1: Continuous
        state1 = solver.initialize(pattern='quadrant', T_hot=600.0)
        for _ in range(total_steps):
            state1 = solver.step(state1, dt)
        
        # Run 2: With restart
        state2 = solver.initialize(pattern='quadrant', T_hot=600.0)
        for _ in range(restart_step):
            state2 = solver.step(state2, dt)
        
        checkpoint = solver.state_to_output(state2, 'state')
        state3 = solver.state_from_checkpoint(checkpoint)
        
        for _ in range(total_steps - restart_step):
            state3 = solver.step(state3, dt)
        
        # Compare
        max_diff = np.max(np.abs(np.array(state1.T) - np.array(state3.T)))
        assert max_diff < 1e-10, f"Restart differs: {max_diff:.2e}"
    
    @pytest.mark.quick
    def test_advection_restart(self, advection_solver_quick):
        """Advection: stop + restart = continuous run."""
        import jax.numpy as jnp
        
        solver = advection_solver_quick
        
        # Compute dt
        state = solver.initialize()
        V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
        V_max = float(jnp.max(V_mag))
        dt = 0.5 * (solver.dx * 6.371e6) / V_max
        
        total_steps = 20
        restart_step = 10
        
        # Run 1: Continuous
        state1 = solver.initialize()
        for _ in range(total_steps):
            state1 = solver.step(state1, dt)
        
        # Run 2: With restart
        state2 = solver.initialize()
        for _ in range(restart_step):
            state2 = solver.step(state2, dt)
        
        checkpoint = solver.state_to_output(state2, 'state')
        state3 = solver.state_from_checkpoint(checkpoint)
        
        for _ in range(total_steps - restart_step):
            state3 = solver.step(state3, dt)
        
        # Compare
        max_diff = np.max(np.abs(np.array(state1.q) - np.array(state3.q)))
        assert max_diff < 1e-10, f"Restart differs: {max_diff:.2e}"


# =============================================================================
# CONSERVATION TESTS
# =============================================================================

class TestConservation:
    """Test conservation properties of solvers."""
    
    @pytest.mark.quick
    def test_diffusion_heat_conservation(self, diffusion_solver_quick):
        """Diffusion conserves heat (within numerical tolerance)."""
        solver = diffusion_solver_quick
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        
        diag0 = solver.get_diagnostics(state)
        heat_initial = diag0['heat_content']
        
        dt = 1800.0
        for _ in range(50):
            state = solver.step(state, dt)
        
        diag_final = solver.get_diagnostics(state)
        heat_error = abs(diag_final['heat_content'] - heat_initial) / heat_initial
        
        # Heat conservation for forward Euler diffusion:
        # - Discrete Laplacian has small metric discretization errors
        # - N=30 (coarse) accumulates ~1e-6 to 1e-5 error over 50 steps
        # - N=120 (fine) shows ~1e-10 error (see test_validation.py)
        # - 1e-5 tolerance is appropriate for quick N=30 test
        assert heat_error < 1e-5, f"Heat conservation error: {heat_error:.2e}"
    
    @pytest.mark.quick
    def test_advection_mass_conservation(self, advection_solver_quick):
        """Advection conserves mass (within tolerance for PLR)."""
        import jax.numpy as jnp
        
        solver = advection_solver_quick
        state = solver.initialize()
        
        V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
        V_max = float(jnp.max(V_mag))
        dt = 0.5 * (solver.dx * 6.371e6) / V_max
        
        diag0 = solver.get_diagnostics(state)
        mass_initial = diag0['mass']
        
        for _ in range(50):
            state = solver.step(state, dt)
        
        diag_final = solver.get_diagnostics(state)
        mass_error = abs(diag_final['mass'] - mass_initial) / mass_initial
        
        # PLR has some numerical diffusion - allow 5%
        assert mass_error < 0.05, f"Mass conservation error: {mass_error:.2e}"


# =============================================================================
# REGRESSION TESTS (SLOW)
# =============================================================================

class TestRegression:
    """Regression tests against reference solutions."""
    
    @pytest.mark.slow
    def test_diffusion_exact_match(self, diffusion_solver_full, diffusion_reference):
        """Diffusion matches reference solution exactly (N=120, 30 days)."""
        solver = diffusion_solver_full
        ref_data = diffusion_reference
        
        days = float(ref_data['days'])
        T_ref = ref_data['T']
        
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        
        dt = 1800.0
        n_steps = int(days * 86400 / dt)
        
        for _ in range(n_steps):
            state = solver.step(state, dt)
        
        T_computed = np.array(state.T)
        max_diff = np.max(np.abs(T_computed - T_ref))
        
        assert max_diff < 1e-10, f"Differs from reference: {max_diff:.2e}"
    
    @pytest.mark.slow
    def test_advection_exact_match(self, advection_solver_full, advection_reference):
        """Advection matches reference solution exactly (N=120, 12 days)."""
        solver = advection_solver_full
        ref_data = advection_reference
        
        days = float(ref_data['days'])
        dt_ref = float(ref_data['dt'])
        q_ref = ref_data['q']
        
        state = solver.initialize(test_case='cosine_bell', u0=None)
        
        n_steps = int(days * 86400 / dt_ref)
        
        for _ in range(n_steps):
            state = solver.step(state, dt_ref)
        
        q_computed = np.array(state.q)
        max_diff = np.max(np.abs(q_computed - q_ref))
        
        assert max_diff < 1e-10, f"Differs from reference: {max_diff:.2e}"


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

class TestDiagnostics:
    """Test diagnostic output correctness."""
    
    @pytest.mark.quick
    def test_diffusion_diagnostics_keys(self, diffusion_solver_quick):
        """Diffusion diagnostics contain expected keys."""
        solver = diffusion_solver_quick
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        diag = solver.get_diagnostics(state)
        
        assert 'T_max' in diag
        assert 'heat_content' in diag
        assert 'time' in diag
        assert 'step' in diag
    
    @pytest.mark.quick
    def test_advection_diagnostics_keys(self, advection_solver_quick):
        """Advection diagnostics contain expected keys."""
        solver = advection_solver_quick
        state = solver.initialize()
        diag = solver.get_diagnostics(state)
        
        assert 'mass' in diag
        assert 'q_max' in diag
        assert 'q_min' in diag
        assert 'face_with_peak' in diag

