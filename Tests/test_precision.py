#!/usr/bin/env python
"""
Precision Comparison Test

Compares solver behavior between float32 and float64 precision.
Key metric: mass/heat conservation should be similar (within tolerance).

This test validates that:
1. Solvers work in both f32 and f64 modes
2. Conservation properties degrade gracefully in f32
3. Results are not garbage (visual verification backup)

Usage:
    python Tests/test_precision.py
    python Tests/test_precision.py --solver diffusion --steps 100
    python Tests/test_precision.py --solver advection --steps 200
"""

import sys
import argparse
import warnings
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress JAX dtype truncation warnings (expected when testing f32 mode)
warnings.filterwarnings('ignore', message='.*dtype.*requested.*not available.*truncated.*')

import jax
import jax.numpy as jnp


def test_diffusion_precision(N=30, steps=100, dt=1800.0, verbose=True):
    """
    Compare diffusion solver in f32 vs f64.
    
    Returns:
        (passed, results_dict)
    """
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    
    config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
    
    results = {}
    
    for precision in ['f64', 'f32']:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing DIFFUSION at {precision}")
            print(f"{'='*60}")
        
        # Enable/disable x64 mode
        if precision == 'f64':
            jax.config.update('jax_enable_x64', True)
        else:
            jax.config.update('jax_enable_x64', False)
        
        # Create solver
        solver = CubedSphereDiffusion(N=N, kappa=5e5, config_file=str(config_path))
        
        # Initialize
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        
        # Get initial diagnostics
        diag0 = solver.get_diagnostics(state)
        heat_initial = diag0['heat_content']
        
        # Run steps
        for _ in range(steps):
            state = solver.step(state, dt)
        
        # Final diagnostics
        diag_final = solver.get_diagnostics(state)
        heat_error = abs(diag_final['heat_content'] - heat_initial) / heat_initial
        
        results[precision] = {
            'heat_initial': heat_initial,
            'heat_final': diag_final['heat_content'],
            'heat_error': heat_error,
            'T_max': diag_final['T_max'],
        }
        
        if verbose:
            print(f"\n  Results ({precision}):")
            print(f"    Heat conservation error: {heat_error:.2e}")
            print(f"    T_max: {diag_final['T_max']:.2f} K")
    
    # Compare: both should be at their respective machine precision
    # f64 machine epsilon ≈ 2.2e-16, f32 machine epsilon ≈ 1.2e-7
    # After N steps, expect O(N * epsilon) accumulated error
    f64_error = results['f64']['heat_error']
    f32_error = results['f32']['heat_error']
    
    # Tolerances based on machine precision (generous bounds)
    # f64: should be < 1e-12 (allows ~10^4 steps of accumulation)
    # f32: should be < 1e-4 (allows ~10^3 steps of accumulation)
    f64_ok = f64_error < 1e-12
    f32_ok = f32_error < 1e-4
    
    passed = f64_ok and f32_ok
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PRECISION COMPARISON")
        print(f"{'='*60}")
        print(f"  f64 heat error: {f64_error:.2e} (expect < 1e-12)")
        print(f"  f32 heat error: {f32_error:.2e} (expect < 1e-4)")
        print(f"  f64 at machine precision: {'✓' if f64_ok else '✗'}")
        print(f"  f32 at machine precision: {'✓' if f32_ok else '✗'}")
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed, results


def test_advection_precision(N=30, steps=100, verbose=True):
    """
    Compare advection solver in f32 vs f64.
    
    Returns:
        (passed, results_dict)
    """
    from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
    
    config_path = Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml'
    
    results = {}
    
    for precision in ['f64', 'f32']:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing ADVECTION at {precision}")
            print(f"{'='*60}")
        
        # Enable/disable x64 mode
        if precision == 'f64':
            jax.config.update('jax_enable_x64', True)
        else:
            jax.config.update('jax_enable_x64', False)
        
        # Create solver
        solver = PLRCubeSphereAdvection(N=N, config_file=str(config_path))
        
        # Initialize
        state = solver.initialize()
        
        # Compute dt from velocity field
        V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
        V_max = float(jnp.max(V_mag))
        dt = 0.5 * (solver.dx * solver.planet.R_sphere) / V_max
        
        # Get initial diagnostics
        diag0 = solver.get_diagnostics(state)
        mass_initial = diag0['mass']
        
        # Run steps
        for _ in range(steps):
            state = solver.step(state, dt)
        
        # Final diagnostics
        diag_final = solver.get_diagnostics(state)
        mass_error = abs(diag_final['mass'] - mass_initial) / mass_initial
        
        results[precision] = {
            'mass_initial': mass_initial,
            'mass_final': diag_final['mass'],
            'mass_error': mass_error,
            'q_max': diag_final['q_max'],
        }
        
        if verbose:
            print(f"\n  Results ({precision}):")
            print(f"    Mass conservation error: {mass_error:.2e}")
            print(f"    q_max: {diag_final['q_max']:.2f}")
    
    # Compare: for advection, error is dominated by numerical scheme (not precision)
    # So f32 and f64 should give similar results
    f64_error = results['f64']['mass_error']
    f32_error = results['f32']['mass_error']
    
    # For advection: 
    # 1. Both should be reasonably small (< 10% for PLR scheme)
    # 2. f32 should not be dramatically worse than f64 (< 10x)
    error_ratio = f32_error / max(f64_error, 1e-15)
    
    both_reasonable = (f64_error < 0.1) and (f32_error < 0.1)
    similar_behavior = error_ratio < 10  # f32 not catastrophically worse
    
    passed = both_reasonable and similar_behavior
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PRECISION COMPARISON")
        print(f"{'='*60}")
        print(f"  f64 mass error: {f64_error:.2e}")
        print(f"  f32 mass error: {f32_error:.2e}")
        print(f"  Ratio (f32/f64): {error_ratio:.1f}x (expect < 10x)")
        print(f"  Both reasonable: {'✓' if both_reasonable else '✗'}")
        print(f"  Similar behavior: {'✓' if similar_behavior else '✗'}")
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    
    return passed, results


def main():
    parser = argparse.ArgumentParser(
        description='Precision comparison test (f32 vs f64)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--solver', type=str, default='both',
                       choices=['diffusion', 'advection', 'both'],
                       help='Which solver to test')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of steps to run')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Grid resolution (N)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    print("\n" + "="*60)
    print("PRECISION COMPARISON TEST (f32 vs f64)")
    print("="*60)
    
    all_passed = True
    
    if args.solver in ['diffusion', 'both']:
        passed, _ = test_diffusion_precision(
            N=args.grid_size, steps=args.steps, verbose=verbose
        )
        all_passed = all_passed and passed
    
    if args.solver in ['advection', 'both']:
        passed, _ = test_advection_precision(
            N=args.grid_size, steps=args.steps, verbose=verbose
        )
        all_passed = all_passed and passed
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL PRECISION TESTS PASSED!")
    else:
        print("❌ SOME PRECISION TESTS FAILED!")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

