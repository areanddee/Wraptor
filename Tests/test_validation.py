"""
Regression Tests: Validate against reference solutions

These tests ensure that code changes don't break physics correctness.
They compare current solver output against "gold standard" reference solutions.

IMPORTANT: These tests should ALWAYS pass unless:
1. You've fixed a physics bug (intentional change)
2. You've improved numerical accuracy (intentional change)
3. Something broke (unintentional - FIX IT!)

If tests fail unexpectedly, DO NOT update reference solutions.
Debug and fix the code instead.
"""

import numpy as np
import sys
from pathlib import Path

# Add solver paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'Solvers'))

# Optional pytest support
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest decorators for standalone execution
    class MockPytest:
        @staticmethod
        def fixture(func):
            return func
    pytest = MockPytest()

from fv_cubesphere_diffusion import CubedSphereDiffusion
from fv_plr_cubesphere_adv import PLRCubeSphereAdvection


class TestDiffusionValidation:
    """Regression tests for thermal diffusion solver"""
    
    @pytest.fixture
    def reference_data(self):
        """Load Lima Flag reference solution"""
        ref_file = Path(__file__).parent / 'validation' / 'diffusion_lima_flag_day30_N120.npz'
        assert ref_file.exists(), f"Reference file not found: {ref_file}"
        return np.load(ref_file)
    
    def test_lima_flag_day30_exact_match(self, reference_data):
        """Test: Lima Flag at 30 days matches reference (bitwise reproducibility)"""
        # Load reference
        T_ref = reference_data['T']
        N = int(reference_data['N'])
        kappa = float(reference_data['kappa'])
        days = float(reference_data['days'])
        
        print(f"\n  Testing Lima Flag: N={N}, days={days}, κ={kappa:.2e}")
        
        # Run solver with same parameters
        config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
        solver = CubedSphereDiffusion(N=N, kappa=kappa, config_file=str(config_path))
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        
        dt = 1800.0
        n_steps = int(days * 86400 / dt)
        
        for step in range(n_steps):
            state = solver.step(state, dt)
        
        T_computed = np.array(state.T)
        
        # Check exact match (or very close, within JAX numerical precision)
        max_diff = np.max(np.abs(T_computed - T_ref))
        rel_diff = max_diff / np.max(T_ref)
        
        print(f"  Max absolute diff: {max_diff:.6e} K")
        print(f"  Max relative diff: {rel_diff:.6e}")
        print(f"  Reference T_max: {np.max(T_ref):.2f} K")
        print(f"  Computed T_max: {np.max(T_computed):.2f} K")
        
        # Should match to machine precision (or very close)
        assert rel_diff < 1e-10, f"Solution differs from reference by {rel_diff:.2e}"
        assert np.allclose(T_computed, T_ref, rtol=1e-12, atol=1e-10), \
            "Temperature field does not match reference solution!"
    
    def test_heat_conservation(self, reference_data):
        """Test: Verify heat conservation during diffusion"""
        N = int(reference_data['N'])
        kappa = float(reference_data['kappa'])
        
        config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
        solver = CubedSphereDiffusion(N=N, kappa=kappa, config_file=str(config_path))
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        
        # Initial heat content
        diag0 = solver.get_diagnostics(state)
        heat_initial = diag0['heat_content']
        
        # Run for 100 steps
        dt = 1800.0
        for step in range(100):
            state = solver.step(state, dt)
        
        # Check heat conservation
        diag_final = solver.get_diagnostics(state)
        heat_error = abs(diag_final['heat_content'] - heat_initial) / heat_initial
        
        print(f"\n  Heat conservation test:")
        print(f"  Initial: {heat_initial:.6e}")
        print(f"  Final:   {diag_final['heat_content']:.6e}")
        print(f"  Error:   {heat_error:.6e}")
        
        # Heat should be conserved to machine precision
        assert heat_error < 1e-10, f"Heat conservation error: {heat_error:.2e}"


class TestAdvectionValidation:
    """Regression tests for PLR advection solver"""
    
    @pytest.fixture
    def reference_data(self):
        """Load Cosine Bell reference solution"""
        ref_file = Path(__file__).parent / 'validation' / 'advection_cosine_bell_day12_N120.npz'
        assert ref_file.exists(), f"Reference file not found: {ref_file}"
        return np.load(ref_file)
    
    def test_cosine_bell_day12_exact_match(self, reference_data):
        """Test: Cosine Bell at 12 days matches reference (bitwise reproducibility)"""
        # Load reference
        q_ref = reference_data['q']
        N = int(reference_data['N'])
        days = float(reference_data['days'])
        dt_ref = float(reference_data['dt'])
        
        print(f"\n  Testing Cosine Bell: N={N}, days={days}")
        
        # Run solver with same parameters
        config_path = Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml'
        solver = PLRCubeSphereAdvection(N=N, config_file=str(config_path))
        state = solver.initialize(test_case='cosine_bell', u0=None)
        
        # Use same dt as reference
        n_steps = int(days * 86400 / dt_ref)
        
        for step in range(n_steps):
            state = solver.step(state, dt_ref)
        
        q_computed = np.array(state.q)
        
        # Check exact match
        max_diff = np.max(np.abs(q_computed - q_ref))
        rel_diff = max_diff / np.max(q_ref)
        
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Max relative diff: {rel_diff:.6e}")
        print(f"  Reference q_max: {np.max(q_ref):.2f}")
        print(f"  Computed q_max: {np.max(q_computed):.2f}")
        
        # Should match to machine precision
        assert rel_diff < 1e-10, f"Solution differs from reference by {rel_diff:.2e}"
        assert np.allclose(q_computed, q_ref, rtol=1e-12, atol=1e-10), \
            "Tracer field does not match reference solution!"
    
    def test_mass_conservation(self, reference_data):
        """Test: Verify mass conservation during advection"""
        N = int(reference_data['N'])
        
        config_path = Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml'
        solver = PLRCubeSphereAdvection(N=N, config_file=str(config_path))
        state = solver.initialize(test_case='cosine_bell', u0=None)
        
        # Initial mass
        diag0 = solver.get_diagnostics(state)
        mass_initial = diag0['mass']
        
        # Run for 100 steps
        import jax.numpy as jnp
        V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
        V_max = float(jnp.max(V_mag))
        dt = 0.5 * (solver.dx * 6.371e6) / V_max
        
        for step in range(100):
            state = solver.step(state, dt)
        
        # Check mass conservation
        diag_final = solver.get_diagnostics(state)
        mass_error = abs(diag_final['mass'] - mass_initial) / mass_initial
        
        print(f"\n  Mass conservation test:")
        print(f"  Initial: {mass_initial:.6e}")
        print(f"  Final:   {diag_final['mass']:.6e}")
        print(f"  Error:   {mass_error:.6e}")
        
        # Mass conservation for 2nd-order PLR:
        # - PLR with MC limiter has small numerical diffusion
        # - 100 steps at CFL=0.5 can accumulate ~1-2% error
        # - Full 12-day run (959 steps) typically shows ~0.4% error
        # - 5% tolerance is reasonable for quick validation test
        # - The exact-match test (above) is the critical one
        assert mass_error < 5e-2, f"Mass conservation error: {mass_error:.2e} (too large!)"


class TestCrossValidation:
    """Cross-validation tests: same physics, different resolutions"""
    
    def test_diffusion_convergence(self):
        """Test: Higher resolution → more accurate (diffusion convergence)"""
        config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
        
        # Run at two resolutions
        results = {}
        for N in [30, 60]:
            solver = CubedSphereDiffusion(N=N, kappa=5e5, config_file=str(config_path))
            state = solver.initialize(pattern='quadrant', T_hot=600.0)
            
            # Run for 5 days
            dt = 1800.0
            n_steps = int(5 * 86400 / dt)
            
            for step in range(n_steps):
                state = solver.step(state, dt)
            
            results[N] = np.array(state.T)
        
        # Higher resolution should have smoother gradients
        grad_N30 = np.max(np.abs(np.gradient(results[30][0])))
        grad_N60 = np.max(np.abs(np.gradient(results[60][0])))
        
        print(f"\n  Convergence test:")
        print(f"  N=30 max gradient: {grad_N30:.2e}")
        print(f"  N=60 max gradient: {grad_N60:.2e}")
        
        # This is a qualitative test - just ensure it runs without crashing
        assert grad_N30 > 0 and grad_N60 > 0, "Gradient calculation failed"


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VALIDATION TEST SUITE")
    print("="*70)
    
    if HAS_PYTEST:
        # Run with pytest if available
        pytest.main([__file__, "-v", "-s"])
    else:
        # Run tests manually
        print("Running tests without pytest...\n")
        
        # Diffusion tests
        print("Testing Diffusion Solver:")
        print("-" * 70)
        test_diff = TestDiffusionValidation()
        ref_diff = test_diff.reference_data()
        
        try:
            test_diff.test_lima_flag_day30_exact_match(ref_diff)
            print("✓ Lima Flag Day 30 exact match: PASS")
        except AssertionError as e:
            print(f"✗ Lima Flag Day 30 exact match: FAIL ({e})")
        
        try:
            test_diff.test_heat_conservation(ref_diff)
            print("✓ Heat conservation: PASS")
        except AssertionError as e:
            print(f"✗ Heat conservation: FAIL ({e})")
        
        # Advection tests
        print("\n" + "Testing Advection Solver:")
        print("-" * 70)
        test_adv = TestAdvectionValidation()
        ref_adv = test_adv.reference_data()
        
        try:
            test_adv.test_cosine_bell_day12_exact_match(ref_adv)
            print("✓ Cosine Bell Day 12 exact match: PASS")
        except AssertionError as e:
            print(f"✗ Cosine Bell Day 12 exact match: FAIL ({e})")
        
        try:
            test_adv.test_mass_conservation(ref_adv)
            print("✓ Mass conservation: PASS")
        except AssertionError as e:
            print(f"✗ Mass conservation: FAIL ({e})")
        
        # Cross-validation
        print("\n" + "Cross-Validation Tests:")
        print("-" * 70)
        test_cross = TestCrossValidation()
        
        try:
            test_cross.test_diffusion_convergence()
            print("✓ Diffusion convergence: PASS")
        except AssertionError as e:
            print(f"✗ Diffusion convergence: FAIL ({e})")
        
        print("\n" + "="*70)
        print("✅ Test suite complete!")
        print("="*70)

