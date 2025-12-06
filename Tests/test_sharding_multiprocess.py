#!/usr/bin/env python
"""
Multi-Process Sharding Test

Runs solver tests with REAL multi-device configurations by spawning
separate Python processes with XLA_FLAGS set BEFORE JAX imports.

Usage:
    python Tests/test_sharding_multiprocess.py                    # Default: 1 device
    python Tests/test_sharding_multiprocess.py --devices 6        # 6 virtual CPUs
    python Tests/test_sharding_multiprocess.py --devices 1,2,3,6  # Full sweep
    python Tests/test_sharding_multiprocess.py --solver advection --devices 1,6

This is the CORRECT way to test multi-device behavior on CPU.
"""

import sys
import os
import argparse
import subprocess
import tempfile
import json
import numpy as np
from pathlib import Path


# Worker script that runs in a subprocess with proper XLA_FLAGS
WORKER_SCRIPT = '''
import os
import sys
import json
import numpy as np

# XLA_FLAGS is already set in environment before this runs
sys.path.insert(0, '{project_root}')

def run_diffusion_test(N, steps, output_file):
    """Run diffusion solver and save results."""
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    
    solver = CubedSphereDiffusion(
        N=N, kappa=5e5,
        config_file='{project_root}/Config/config_diffusion_no_sharding.yaml'
    )
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    dt = 1800.0
    for step in range(steps):
        state = solver.step(state, dt)
    
    # Save results
    np.savez(output_file, T=np.array(state.T), time=state.time, step=state.step)


def run_advection_test(N, steps, output_file):
    """Run advection solver and save results."""
    import jax.numpy as jnp
    from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
    
    solver = PLRCubeSphereAdvection(
        N=N,
        config_file='{project_root}/Config/config_plr_advection_no_sharding.yaml'
    )
    state = solver.initialize()
    
    V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
    V_max = float(jnp.max(V_mag))
    dt = 0.5 * (solver.dx * 6.371e6) / V_max
    
    for step in range(steps):
        state = solver.step(state, dt)
    
    # Save results
    np.savez(output_file, q=np.array(state.q), time=state.time, step=state.step)


if __name__ == "__main__":
    solver_type = sys.argv[1]
    N = int(sys.argv[2])
    steps = int(sys.argv[3])
    output_file = sys.argv[4]
    
    if solver_type == 'diffusion':
        run_diffusion_test(N, steps, output_file)
    else:
        run_advection_test(N, steps, output_file)
'''


def run_with_devices(num_devices, solver_type, N, steps, project_root, verbose=True):
    """
    Run solver in subprocess with specific device count.
    
    Returns path to output file containing results.
    """
    if verbose:
        print(f"  Running with {num_devices} device(s)...", end=" ", flush=True)
    
    # Create temporary files
    script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    output_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    output_file.close()
    
    # Write worker script
    script_content = WORKER_SCRIPT.format(project_root=project_root)
    script_file.write(script_content)
    script_file.close()
    
    try:
        # Set XLA_FLAGS in environment BEFORE starting subprocess
        env = os.environ.copy()
        env['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_devices}'
        
        # Run subprocess
        result = subprocess.run(
            [sys.executable, script_file.name, solver_type, str(N), str(steps), output_file.name],
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"✗")
                print(f"    STDERR: {result.stderr[:200]}")
            return None
        
        if verbose:
            print("✓")
        
        return output_file.name
        
    except subprocess.TimeoutExpired:
        if verbose:
            print("✗ (timeout)")
        return None
    finally:
        os.unlink(script_file.name)


def compare_results(file1, file2, solver_type):
    """Compare two result files and return max difference."""
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    if solver_type == 'diffusion':
        diff = np.max(np.abs(data1['T'] - data2['T']))
    else:
        diff = np.max(np.abs(data1['q'] - data2['q']))
    
    return diff


def run_sharding_test(solver_type='diffusion', device_counts=None, N=30, steps=20, verbose=True):
    """
    Run sharding comparison test.
    
    Args:
        solver_type: 'diffusion' or 'advection'
        device_counts: List of device counts to test (e.g., [1, 6] or [1, 2, 3, 6])
        N: Grid resolution
        steps: Number of steps to run
        verbose: Print detailed output
    
    Returns:
        (all_passed, results): Boolean and dict of results
    """
    if device_counts is None:
        device_counts = [1]
    
    project_root = str(Path(__file__).parent.parent)
    
    if verbose:
        print("\n" + "="*70)
        print("MULTI-PROCESS SHARDING TEST")
        print("="*70)
        print(f"Solver: {solver_type}")
        print(f"Resolution: N={N}")
        print(f"Steps: {steps}")
        print(f"Device counts: {device_counts}")
        print("="*70)
        print("\n[Phase 1] Running solver with each device count...")
    
    # Run with each device count
    result_files = {}
    for num_devices in device_counts:
        output_file = run_with_devices(
            num_devices, solver_type, N, steps, project_root, verbose
        )
        if output_file is None:
            if verbose:
                print(f"\n❌ FAILED: Could not run with {num_devices} device(s)")
            return False, {}
        result_files[num_devices] = output_file
    
    # Compare results
    if len(device_counts) > 1:
        if verbose:
            print(f"\n[Phase 2] Comparing results (baseline: {device_counts[0]} device(s))...")
        
        baseline = device_counts[0]
        all_passed = True
        max_diffs = {}
        
        for num_devices in device_counts[1:]:
            diff = compare_results(
                result_files[baseline], 
                result_files[num_devices],
                solver_type
            )
            max_diffs[num_devices] = diff
            passed = diff < 1e-10
            
            if not passed:
                all_passed = False
            
            if verbose:
                status = "✓" if passed else "✗"
                print(f"  {baseline}-dev vs {num_devices}-dev: max_diff = {diff:.2e} {status}")
        
        # Cleanup
        for f in result_files.values():
            try:
                os.unlink(f)
            except:
                pass
        
        if verbose:
            print("\n" + "="*70)
            if all_passed:
                print("✅ PASSED: All device configurations produce identical results!")
            else:
                print("❌ FAILED: Some configurations differ!")
            print("="*70 + "\n")
        
        return all_passed, max_diffs
    
    else:
        # Single device test - just verify it runs
        data = np.load(result_files[device_counts[0]])
        
        if verbose:
            if solver_type == 'diffusion':
                print(f"\n  Results: T_max = {np.max(data['T']):.6f} K")
            else:
                print(f"\n  Results: q_max = {np.max(data['q']):.6f}")
            print("\n" + "="*70)
            print("✅ PASSED: Single-device test completed successfully!")
            print("="*70 + "\n")
        
        # Cleanup
        os.unlink(result_files[device_counts[0]])
        
        return True, {}


def main():
    parser = argparse.ArgumentParser(
        description='Multi-process sharding test with proper XLA_FLAGS',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--solver', type=str, default='diffusion',
                       choices=['diffusion', 'advection'],
                       help='Solver to test')
    parser.add_argument('--devices', type=str, default='1',
                       help='Device count(s): "1", "6", or "1,2,3,6"')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of steps to run')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Grid resolution (N)')
    parser.add_argument('--quiet', action='store_true',
                       help='Only print pass/fail')
    
    args = parser.parse_args()
    
    device_counts = [int(x.strip()) for x in args.devices.split(',')]
    
    passed, results = run_sharding_test(
        solver_type=args.solver,
        device_counts=device_counts,
        N=args.grid_size,
        steps=args.steps,
        verbose=not args.quiet
    )
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

