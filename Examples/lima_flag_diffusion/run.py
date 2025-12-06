#!/usr/bin/env python
"""
Lima Flag Diffusion Example

A thin wrapper demonstrating how to use the diffusion solver from the Framework.
This is a tutorial example - the heavy lifting is in the Solver.

Usage:
    cd Examples/lima_flag_diffusion
    python run.py                           # Default: N=60, 10 days
    python run.py --grid-size 120 --days 30 # High resolution, 30 days
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion


def main():
    parser = argparse.ArgumentParser(
        description='Lima Flag Diffusion Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-size', '-N', type=int, default=60,
                        help='Grid resolution (N×N cells per face)')
    parser.add_argument('--days', type=float, default=10.0,
                        help='Simulation length in days')
    parser.add_argument('--kappa', type=float, default=5e5,
                        help='Diffusion coefficient [m²/s]')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("="*70)
    print("LIMA FLAG DIFFUSION EXAMPLE")
    print("="*70)
    
    config_file = PROJECT_ROOT / 'Config' / 'config_diffusion_no_sharding.yaml'
    
    # Create solver
    solver = CubedSphereDiffusion(
        N=args.grid_size,
        kappa=args.kappa,
        config_file=str(config_file)
    )
    
    # Initialize with Lima Flag pattern
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    # =========================================================================
    # TIME INTEGRATION
    # =========================================================================
    dt = 1800.0  # 30 minutes
    total_time = args.days * 86400.0
    n_steps = int(total_time / dt)
    
    print(f"\nRunning {n_steps} steps ({args.days} days)...")
    print(f"  dt = {dt:.0f}s, N = {args.grid_size}")
    
    # Get initial diagnostics
    diag0 = solver.get_diagnostics(state)
    print(f"\nInitial state:")
    print(f"  T_max = {diag0['T_max']:.1f} K")
    print(f"  Heat  = {diag0['heat_content']:.4e}")
    
    # Run simulation
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        # Progress every 10%
        if (step + 1) % (n_steps // 10) == 0:
            pct = 100 * (step + 1) // n_steps
            diag = solver.get_diagnostics(state)
            print(f"  [{pct:3d}%] Day {diag['time']/86400:.1f}: T_max = {diag['T_max']:.1f} K")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    diag_final = solver.get_diagnostics(state)
    heat_error = abs(diag_final['heat_content'] - diag0['heat_content']) / diag0['heat_content']
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  T_max: {diag0['T_max']:.1f} K → {diag_final['T_max']:.1f} K")
    print(f"  Heat conservation error: {heat_error:.2e}")
    print(f"{'='*70}")
    
    # Save final state (optional)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / 'final_state.npz',
        T=np.array(state.T),
        time=state.time,
        step=state.step,
        N=args.grid_size,
        days=args.days,
        kappa=args.kappa
    )
    print(f"\n✓ Final state saved to {output_dir / 'final_state.npz'}")


if __name__ == "__main__":
    main()

