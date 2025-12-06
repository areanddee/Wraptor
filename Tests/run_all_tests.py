#!/usr/bin/env python
"""
Test Dashboard - Run All Tests with Clean Summary

Usage:
    python Tests/run_all_tests.py              # Run all quick tests
    python Tests/run_all_tests.py --full       # Include long-running tests
    python Tests/run_all_tests.py --sharding 1 # Single device sharding test
    python Tests/run_all_tests.py --sharding 6 # 6-device sharding test  
    python Tests/run_all_tests.py --sharding sweep  # Full sweep (1,2,3,6)
    python Tests/run_all_tests.py --verbose    # Show detailed output

This provides a clean pass/fail dashboard instead of scrolling through pages.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(name, test_func, verbose=False):
    """Run a single test and return (passed, duration, error_msg)."""
    start = time.time()
    try:
        if verbose:
            result = test_func()
        else:
            # Suppress output
            import io
            import contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                result = test_func()
        
        duration = time.time() - start
        
        if result is None or result == True:
            return True, duration, None
        else:
            return False, duration, str(result)
            
    except AssertionError as e:
        duration = time.time() - start
        return False, duration, str(e)
    except Exception as e:
        duration = time.time() - start
        return False, duration, f"{type(e).__name__}: {e}"


def test_references():
    """Quick: Verify reference files exist."""
    import numpy as np
    
    ref_dir = Path(__file__).parent / 'validation'
    
    diff_ref = ref_dir / 'diffusion_lima_flag_day30_N120.npz'
    adv_ref = ref_dir / 'advection_cosine_bell_day12_N120.npz'
    
    assert diff_ref.exists(), f"Missing: {diff_ref}"
    assert adv_ref.exists(), f"Missing: {adv_ref}"
    
    # Quick load test
    np.load(diff_ref)
    np.load(adv_ref)
    
    return True


def test_diffusion_framework():
    """Quick: Diffusion solver implements Framework interface."""
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    from Framework.solver_interface import validate_solver_interface
    
    solver = CubedSphereDiffusion(
        N=30, kappa=5e5,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml')
    )
    validate_solver_interface(solver)
    return True


def test_advection_framework():
    """Quick: Advection solver implements Framework interface."""
    from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
    from Framework.solver_interface import validate_solver_interface
    
    solver = PLRCubeSphereAdvection(
        N=30,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml')
    )
    validate_solver_interface(solver)
    return True


def test_diffusion_restart():
    """Medium: Diffusion restart produces identical results."""
    from Tests.test_restart_exactness import run_restart_test
    passed, max_diff = run_restart_test(
        solver_type='diffusion',
        total_steps=20,
        restart_step=10,
        N=30,
        verbose=False
    )
    assert passed, f"Restart diff: {max_diff:.2e}"
    return True


def test_advection_restart():
    """Medium: Advection restart produces identical results."""
    from Tests.test_restart_exactness import run_restart_test
    passed, max_diff = run_restart_test(
        solver_type='advection',
        total_steps=20,
        restart_step=10,
        N=30,
        verbose=False
    )
    assert passed, f"Restart diff: {max_diff:.2e}"
    return True


def test_diffusion_exact_match():
    """Long: Diffusion matches reference solution (30 days, N=120)."""
    import numpy as np
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    
    ref_file = Path(__file__).parent / 'validation' / 'diffusion_lima_flag_day30_N120.npz'
    ref_data = np.load(ref_file)
    
    N = int(ref_data['N'])
    kappa = float(ref_data['kappa'])
    days = float(ref_data['days'])
    T_ref = ref_data['T']
    
    solver = CubedSphereDiffusion(
        N=N, kappa=kappa,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml')
    )
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    dt = 1800.0
    n_steps = int(days * 86400 / dt)
    
    for step in range(n_steps):
        state = solver.step(state, dt)
    
    T_computed = np.array(state.T)
    max_diff = np.max(np.abs(T_computed - T_ref))
    
    assert max_diff < 1e-10, f"Diff from reference: {max_diff:.2e}"
    return True


def test_advection_exact_match():
    """Long: Advection matches reference solution (12 days, N=120)."""
    import numpy as np
    from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
    
    ref_file = Path(__file__).parent / 'validation' / 'advection_cosine_bell_day12_N120.npz'
    ref_data = np.load(ref_file)
    
    N = int(ref_data['N'])
    days = float(ref_data['days'])
    dt_ref = float(ref_data['dt'])
    q_ref = ref_data['q']
    
    solver = PLRCubeSphereAdvection(
        N=N,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml')
    )
    state = solver.initialize(test_case='cosine_bell', u0=None)
    
    n_steps = int(days * 86400 / dt_ref)
    
    for step in range(n_steps):
        state = solver.step(state, dt_ref)
    
    q_computed = np.array(state.q)
    max_diff = np.max(np.abs(q_computed - q_ref))
    
    assert max_diff < 1e-10, f"Diff from reference: {max_diff:.2e}"
    return True


def test_precision_diffusion():
    """Quick: Compare diffusion in f32 vs f64."""
    from Tests.test_precision import test_diffusion_precision
    passed, _ = test_diffusion_precision(N=30, steps=50, verbose=False)
    assert passed, "Precision test failed (f32 vs f64)"
    return True


def test_precision_advection():
    """Quick: Compare advection in f32 vs f64."""
    from Tests.test_precision import test_advection_precision
    passed, _ = test_advection_precision(N=30, steps=50, verbose=False)
    assert passed, "Precision test failed (f32 vs f64)"
    return True


# Test definitions: (name, function, is_quick)
ALL_TESTS = [
    ("Reference files exist", test_references, True),
    ("Diffusion Framework interface", test_diffusion_framework, True),
    ("Advection Framework interface", test_advection_framework, True),
    ("Diffusion restart exactness", test_diffusion_restart, True),
    ("Advection restart exactness", test_advection_restart, True),
    ("Diffusion precision (f32 vs f64)", test_precision_diffusion, True),
    ("Advection precision (f32 vs f64)", test_precision_advection, True),
    ("Diffusion exact match (N=120, 30 days)", test_diffusion_exact_match, False),
    ("Advection exact match (N=120, 12 days)", test_advection_exact_match, False),
]


def run_sharding_test(device_counts, verbose=True):
    """Run multi-process sharding tests."""
    from Tests.test_sharding_multiprocess import run_sharding_test as mp_sharding_test
    
    results = {}
    all_passed = True
    
    for solver in ['diffusion', 'advection']:
        if verbose:
            print(f"\n  Testing {solver} with {device_counts} device(s)...", end=" ", flush=True)
        
        passed, diffs = mp_sharding_test(
            solver_type=solver,
            device_counts=device_counts if isinstance(device_counts, list) else [device_counts],
            N=30,
            steps=20,
            verbose=False
        )
        
        results[solver] = passed
        if not passed:
            all_passed = False
        
        if verbose:
            print("✓" if passed else "✗")
    
    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description='Run all tests with clean dashboard output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--full', action='store_true',
                       help='Include long-running regression tests')
    parser.add_argument('--sharding', type=str, default=None,
                       help='Sharding test: "1", "6", or "sweep" (1,2,3,6)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output from each test')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("JAXSTREAM2 TEST DASHBOARD")
    print("="*70)
    
    # Handle sharding tests separately
    if args.sharding:
        if args.sharding == 'sweep':
            device_counts = [1, 2, 3, 6]
            print(f"Running sharding sweep test ({device_counts})")
        else:
            device_counts = [int(args.sharding)]
            print(f"Running sharding test with {device_counts[0]} device(s)")
        
        print("="*70)
        
        passed, results = run_sharding_test(device_counts, verbose=True)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for solver, p in results.items():
            status = "✓" if p else "✗"
            print(f"  {solver}: {status}")
        print("="*70)
        if passed:
            print("✅ ALL SHARDING TESTS PASSED!")
        else:
            print("❌ SOME SHARDING TESTS FAILED!")
        print("="*70 + "\n")
        
        sys.exit(0 if passed else 1)
    
    # Regular tests
    if args.full:
        tests = ALL_TESTS
        print(f"Running {len(tests)} tests (including long runs)\n")
    else:
        tests = [(n, f, q) for n, f, q in ALL_TESTS if q]
        print(f"Running {len(tests)} quick tests\n")
        print("(Use --full to include long-running regression tests)\n")
    
    results = []
    total_time = 0
    
    for name, func, is_quick in tests:
        tag = "[quick]" if is_quick else "[long] "
        print(f"  {tag} {name}...", end=" ", flush=True)
        
        passed, duration, error = run_test(name, func, args.verbose)
        total_time += duration
        results.append((name, passed, duration, error))
        
        if passed:
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s)")
            if error and not args.verbose:
                print(f"         └─ {error[:60]}...")
    
    # Summary
    passed_count = sum(1 for _, p, _, _ in results if p)
    failed_count = len(results) - passed_count
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Total tests: {len(results)}")
    print(f"  Passed:      {passed_count} ✓")
    print(f"  Failed:      {failed_count} ✗")
    print(f"  Time:        {total_time:.1f}s")
    print("="*70)
    
    if failed_count == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED:")
        for name, passed, _, error in results:
            if not passed:
                print(f"  • {name}")
                if error:
                    print(f"    └─ {error}")
    
    print("="*70)
    
    if failed_count > 0:
        print("\nTo investigate a failure, run the specific test with verbose output:")
        print("  python Tests/test_validation.py  # For exact match tests")
        print("  python Tests/test_restart_exactness.py --solver advection")
    
    print()
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()

