"""
Sharding Reproducibility Sweep Test

Verifies that: same physics with 1, 2, 3, 6 devices produces identical results

This test runs the diffusion solver with different device counts and compares:
- All runs must produce bitwise-identical (or near-identical) results
- Validates that sharding doesn't introduce numerical differences

Usage:
    python Tests/test_sharding_sweep.py
    
    # With options:
    python Tests/test_sharding_sweep.py --solver diffusion --steps 50 --devices 1,2,3,6
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import yaml

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sharding_config(num_devices, output_dir):
    """Create a temporary config file for specific device count."""
    config = {
        'parallelization': {
            'enable_sharding': num_devices > 1,
            'tiles_per_edge': 1,
            'device_type': 'cpu',
            'num_devices': num_devices
        }
    }
    
    config_path = output_dir / f'config_sharding_{num_devices}dev.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


def run_with_devices(num_devices, N, steps, dt, config_dir, verbose=True):
    """
    Run diffusion solver with specific device count.
    
    Note: For CPU testing, XLA_FLAGS must be set BEFORE importing JAX.
    Since JAX is already imported, we set it for the subprocess behavior,
    but for this single-process test, all runs use the same JAX instance.
    
    For true multi-device testing, we'd need separate processes.
    This test validates the LOGIC works; true device parallelism
    requires actual multi-GPU hardware.
    """
    if verbose:
        print(f"\n  Running with {num_devices} device(s)...")
    
    # Set XLA flags for virtual devices (must be done before JAX import in real use)
    # Note: This has limited effect since JAX is already imported
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'
    
    # Create config
    config_path = create_sharding_config(num_devices, config_dir)
    
    # Import solver (JAX already initialized, so sharding may not fully work)
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    
    # Create solver with sharding config
    solver = CubedSphereDiffusion(N=N, kappa=5e5, config_file=config_path)
    
    # Initialize
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    # Run
    for step in range(steps):
        state = solver.step(state, dt)
    
    T_final = np.array(state.T)
    T_max = np.max(T_final)
    
    if verbose:
        print(f"    T_max: {T_max:.6f} K")
    
    return T_final


def run_sharding_sweep(solver_type='diffusion', steps=50, device_counts=None,
                       N=30, verbose=True):
    """
    Test that different device counts produce identical results.
    
    Args:
        solver_type: 'diffusion' (advection to be added)
        steps: Number of steps to run
        device_counts: List of device counts to test (default: [1, 2, 3, 6])
        N: Grid resolution
        verbose: Print detailed output
    
    Returns:
        (all_passed, results): Boolean and dict of results
    """
    if device_counts is None:
        device_counts = [1, 2, 3, 6]
    
    if verbose:
        print("="*70)
        print("SHARDING REPRODUCIBILITY SWEEP")
        print("="*70)
        print(f"Solver: {solver_type}")
        print(f"Resolution: N={N}")
        print(f"Steps: {steps}")
        print(f"Device counts to test: {device_counts}")
        print("="*70)
        print("\n⚠️  Note: True multi-device testing requires multi-GPU hardware.")
        print("   This test validates sharding LOGIC on virtual CPU devices.")
        print("   XLA creates virtual devices but JAX initialization limits effect.")
    
    # Create temporary config directory
    import tempfile
    config_dir = Path(tempfile.mkdtemp(prefix='jaxstream_sharding_test_'))
    
    try:
        dt = 1800.0  # 30 min
        results = {}
        
        # Run with each device count
        for num_devices in device_counts:
            T_final = run_with_devices(
                num_devices, N, steps, dt, config_dir, verbose
            )
            results[num_devices] = T_final
        
        # ====================================================================
        # COMPARE ALL RESULTS
        # ====================================================================
        if verbose:
            print("\n" + "="*70)
            print("COMPARISON (all runs vs. 1-device baseline)")
            print("="*70)
        
        baseline = results[device_counts[0]]
        all_passed = True
        max_diffs = {}
        
        for num_devices in device_counts[1:]:
            T = results[num_devices]
            max_diff = np.max(np.abs(T - baseline))
            rel_diff = max_diff / np.max(baseline) if np.max(baseline) > 0 else max_diff
            max_diffs[num_devices] = max_diff
            
            # Allow small tolerance for floating point (should be near-exact)
            passed = max_diff < 1e-10
            
            if not passed:
                all_passed = False
            
            if verbose:
                status = "✓" if passed else "✗"
                print(f"  {device_counts[0]}-dev vs {num_devices}-dev: " +
                      f"max_diff={max_diff:.2e} {status}")
        
        if verbose:
            print("="*70)
            if all_passed:
                print("✅ PASSED: All device configurations produce identical results!")
            else:
                print("❌ FAILED: Some configurations differ!")
            print("="*70)
            
            print("\nNOTE: If test shows differences, this could be due to:")
            print("  1. JAX already initialized (XLA_FLAGS set too late)")
            print("  2. Different JIT compilation paths with sharding")
            print("  3. Actual sharding bugs (most serious)")
            print("\nFor production testing on multi-GPU systems:")
            print("  - Run each device count in a SEPARATE Python process")
            print("  - Set XLA_FLAGS BEFORE Python starts")
            print("  - Compare saved outputs")
        
        return all_passed, results
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='Test that sharding produces identical results across device counts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--solver', type=str, default='diffusion',
                       choices=['diffusion'],  # Add 'advection' when ready
                       help='Solver to test')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of steps to run')
    parser.add_argument('--devices', type=str, default='1,2,3,6',
                       help='Comma-separated list of device counts to test')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Grid resolution (N)')
    parser.add_argument('--quiet', action='store_true',
                       help='Only print pass/fail')
    
    args = parser.parse_args()
    
    device_counts = [int(x) for x in args.devices.split(',')]
    
    passed, results = run_sharding_sweep(
        solver_type=args.solver,
        steps=args.steps,
        device_counts=device_counts,
        N=args.grid_size,
        verbose=not args.quiet
    )
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

