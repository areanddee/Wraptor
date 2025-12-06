"""
Restart Exactness Test

Verifies that: stop + restart = continuous run (bitwise identical)

This test runs the diffusion solver in two modes:
1. Continuous: Run 100 steps straight through
2. Restart: Run 50 steps, checkpoint, restart, run 50 more steps

The final states must match exactly (bitwise).

Usage:
    python Tests/test_restart_exactness.py
    
    # With options:
    python Tests/test_restart_exactness.py --solver diffusion --steps 100 --restart-at 50
"""

import sys
import argparse
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion, DiffusionState
from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection, AdvectionState


def run_restart_test(solver_type='diffusion', total_steps=100, restart_step=50, 
                     N=30, verbose=True):
    """
    Test that restart produces identical results to continuous run.
    
    Args:
        solver_type: 'diffusion' or 'advection'
        total_steps: Total number of steps
        restart_step: Step at which to checkpoint and restart
        N: Grid resolution
        verbose: Print detailed output
    
    Returns:
        (passed, max_diff): Boolean success and max difference
    """
    if verbose:
        print("="*70)
        print("RESTART EXACTNESS TEST")
        print("="*70)
        print(f"Solver: {solver_type}")
        print(f"Resolution: N={N}")
        print(f"Total steps: {total_steps}")
        print(f"Restart at step: {restart_step}")
        print("="*70)
    
    # Create temporary checkpoint directory
    checkpoint_dir = tempfile.mkdtemp(prefix='jaxstream_restart_test_')
    
    try:
        if solver_type == 'advection':
            return run_advection_restart_test(total_steps, restart_step, N, verbose)
        
        # ====================================================================
        # DIFFUSION: RUN 1 - CONTINUOUS (no restart)
        # ====================================================================
        if verbose:
            print("\n[RUN 1] Continuous run (no restart)...")
        
        solver1 = CubedSphereDiffusion(
            N=N, 
            kappa=5e5,
            config_file=str(Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml')
        )
        state1 = solver1.initialize(pattern='quadrant', T_hot=600.0)
        
        dt = 1800.0  # 30 min
        
        # Run all steps
        for step in range(total_steps):
            state1 = solver1.step(state1, dt)
        
        T_continuous = np.array(state1.T)
        
        if verbose:
            print(f"   ✓ Completed {total_steps} steps")
            print(f"   Final T_max: {np.max(T_continuous):.6f} K")
        
        # ====================================================================
        # RUN 2: WITH RESTART
        # ====================================================================
        if verbose:
            print(f"\n[RUN 2] Run with restart at step {restart_step}...")
        
        solver2 = CubedSphereDiffusion(
            N=N, 
            kappa=5e5,
            config_file=str(Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml')
        )
        state2 = solver2.initialize(pattern='quadrant', T_hot=600.0)
        
        # Run until restart point
        for step in range(restart_step):
            state2 = solver2.step(state2, dt)
        
        if verbose:
            print(f"   Reached step {restart_step}, creating checkpoint...")
        
        # Simulate checkpoint: save state to numpy
        checkpoint_data = solver2.state_to_output(state2, 'state')
        
        if verbose:
            print(f"   Checkpoint saved (T_max: {np.max(checkpoint_data['T']):.6f} K)")
        
        # Simulate restart: create new solver and restore
        solver3 = CubedSphereDiffusion(
            N=N, 
            kappa=5e5,
            config_file=str(Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml')
        )
        
        # Restore state
        state3 = solver3.state_from_checkpoint(checkpoint_data)
        
        if verbose:
            print(f"   Restored from checkpoint (T_max: {np.max(state3.T):.6f} K)")
        
        # Continue running
        remaining_steps = total_steps - restart_step
        for step in range(remaining_steps):
            state3 = solver3.step(state3, dt)
        
        T_restart = np.array(state3.T)
        
        if verbose:
            print(f"   ✓ Completed {remaining_steps} more steps")
            print(f"   Final T_max: {np.max(T_restart):.6f} K")
        
        # ====================================================================
        # COMPARE RESULTS
        # ====================================================================
        max_diff = np.max(np.abs(T_continuous - T_restart))
        rel_diff = max_diff / np.max(T_continuous) if np.max(T_continuous) > 0 else max_diff
        
        if verbose:
            print("\n" + "="*70)
            print("COMPARISON")
            print("="*70)
            print(f"Max absolute difference: {max_diff:.6e} K")
            print(f"Max relative difference: {rel_diff:.6e}")
        
        # Check for exact match (or very close due to JAX floating point)
        passed = max_diff < 1e-10
        
        if verbose:
            print("="*70)
            if passed:
                print("✅ PASSED: Restart produces identical results!")
            else:
                print("❌ FAILED: Results differ after restart!")
                print(f"   Tolerance: 1e-10, Got: {max_diff:.2e}")
            print("="*70)
        
        return passed, max_diff
        
    finally:
        # Clean up
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def run_advection_restart_test(total_steps, restart_step, N, verbose):
    """Run restart test for advection solver."""
    import jax.numpy as jnp
    
    # ========================================================================
    # ADVECTION: RUN 1 - CONTINUOUS (no restart)
    # ========================================================================
    if verbose:
        print("\n[RUN 1] Continuous run (no restart)...")
    
    solver1 = PLRCubeSphereAdvection(
        N=N,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml')
    )
    state1 = solver1.initialize()
    
    # Compute timestep
    V_mag = jnp.sqrt(state1.Vx**2 + state1.Vy**2 + state1.Vz**2)
    V_max = float(jnp.max(V_mag))
    dt = 0.5 * (solver1.dx * 6.371e6) / V_max
    
    # Run all steps
    for step in range(total_steps):
        state1 = solver1.step(state1, dt)
    
    q_continuous = np.array(state1.q)
    
    if verbose:
        print(f"   ✓ Completed {total_steps} steps")
        print(f"   Final q_max: {np.max(q_continuous):.6f}")
    
    # ========================================================================
    # ADVECTION: RUN 2 - WITH RESTART
    # ========================================================================
    if verbose:
        print(f"\n[RUN 2] Run with restart at step {restart_step}...")
    
    solver2 = PLRCubeSphereAdvection(
        N=N,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml')
    )
    state2 = solver2.initialize()
    
    # Run until restart point
    for step in range(restart_step):
        state2 = solver2.step(state2, dt)
    
    if verbose:
        print(f"   Reached step {restart_step}, creating checkpoint...")
    
    # Simulate checkpoint
    checkpoint_data = solver2.state_to_output(state2, 'state')
    
    if verbose:
        print(f"   Checkpoint saved (q_max: {np.max(checkpoint_data['q']):.6f})")
    
    # Simulate restart
    solver3 = PLRCubeSphereAdvection(
        N=N,
        config_file=str(Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml')
    )
    
    state3 = solver3.state_from_checkpoint(checkpoint_data)
    
    if verbose:
        print(f"   Restored from checkpoint (q_max: {np.max(state3.q):.6f})")
    
    # Continue running
    remaining_steps = total_steps - restart_step
    for step in range(remaining_steps):
        state3 = solver3.step(state3, dt)
    
    q_restart = np.array(state3.q)
    
    if verbose:
        print(f"   ✓ Completed {remaining_steps} more steps")
        print(f"   Final q_max: {np.max(q_restart):.6f}")
    
    # ========================================================================
    # COMPARE RESULTS
    # ========================================================================
    max_diff = np.max(np.abs(q_continuous - q_restart))
    rel_diff = max_diff / np.max(q_continuous) if np.max(q_continuous) > 0 else max_diff
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Max relative difference: {rel_diff:.6e}")
    
    passed = max_diff < 1e-10
    
    if verbose:
        print("="*70)
        if passed:
            print("✅ PASSED: Restart produces identical results!")
        else:
            print("❌ FAILED: Results differ after restart!")
            print(f"   Tolerance: 1e-10, Got: {max_diff:.2e}")
        print("="*70)
    
    return passed, max_diff


def main():
    parser = argparse.ArgumentParser(
        description='Test that checkpoint/restart produces identical results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--solver', type=str, default='diffusion',
                       choices=['diffusion', 'advection'],
                       help='Solver to test')
    parser.add_argument('--steps', type=int, default=100,
                       help='Total number of steps')
    parser.add_argument('--restart-at', type=int, default=50,
                       help='Step at which to checkpoint and restart')
    parser.add_argument('--grid-size', type=int, default=30,
                       help='Grid resolution (N)')
    parser.add_argument('--quiet', action='store_true',
                       help='Only print pass/fail')
    
    args = parser.parse_args()
    
    passed, max_diff = run_restart_test(
        solver_type=args.solver,
        total_steps=args.steps,
        restart_step=args.restart_at,
        N=args.grid_size,
        verbose=not args.quiet
    )
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

