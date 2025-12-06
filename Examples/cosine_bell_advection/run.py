#!/usr/bin/env python
"""
Cosine Bell Advection Example

A thin wrapper demonstrating how to use the PLR advection solver.
This runs Williamson Test Case 1: solid-body rotation of a cosine bell.

Usage:
    cd Examples/cosine_bell_advection
    python run.py                           # Default: N=60, 12 days (1 orbit)
    python run.py --grid-size 120 --days 24 # High resolution, 2 orbits
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection


def main():
    parser = argparse.ArgumentParser(
        description='Cosine Bell Advection Demo (Williamson Test Case 1)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-size', '-N', type=int, default=60,
                        help='Grid resolution (N×N cells per face)')
    parser.add_argument('--days', type=float, default=12.0,
                        help='Simulation length in days (12 days = 1 orbit)')
    parser.add_argument('--cfl', type=float, default=0.5,
                        help='CFL number for timestep')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("="*70)
    print("COSINE BELL ADVECTION EXAMPLE")
    print("="*70)
    print("Williamson Test Case 1: Solid-body rotation")
    
    config_file = PROJECT_ROOT / 'Config' / 'config_plr_advection_no_sharding.yaml'
    
    # Create solver
    solver = PLRCubeSphereAdvection(
        N=args.grid_size,
        config_file=str(config_file)
    )
    
    # Initialize with cosine bell
    state = solver.initialize(test_case='cosine_bell', u0=None)
    
    # =========================================================================
    # TIME INTEGRATION
    # =========================================================================
    # Compute timestep from velocity field
    R_SPHERE = 6.371e6
    V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
    V_max = float(jnp.max(V_mag))
    dt = args.cfl * (solver.dx * R_SPHERE) / V_max
    
    total_time = args.days * 86400.0
    n_steps = int(total_time / dt)
    
    print(f"\nRunning {n_steps} steps ({args.days} days)...")
    print(f"  dt = {dt:.0f}s, CFL = {args.cfl}, N = {args.grid_size}")
    
    # Get initial diagnostics
    diag0 = solver.get_diagnostics(state)
    print(f"\nInitial state:")
    print(f"  q_max = {diag0['q_max']:.1f}")
    print(f"  Mass  = {diag0['mass']:.4e}")
    
    # Run simulation
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        # Progress every 10%
        if (step + 1) % (n_steps // 10) == 0:
            pct = 100 * (step + 1) // n_steps
            diag = solver.get_diagnostics(state)
            print(f"  [{pct:3d}%] Day {state.time/86400:.1f}: q_max = {diag['q_max']:.1f}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    diag_final = solver.get_diagnostics(state)
    mass_error = abs(diag_final['mass'] - diag0['mass']) / diag0['mass']
    peak_preserved = diag_final['q_max'] / diag0['q_max'] * 100
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  q_max: {diag0['q_max']:.1f} → {diag_final['q_max']:.1f} ({peak_preserved:.1f}% preserved)")
    print(f"  Mass conservation error: {mass_error:.2e}")
    print(f"  Peak location: Face {diag_final['face_with_peak']}")
    print(f"{'='*70}")
    
    # Save final state (optional)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / 'final_state.npz',
        q=np.array(state.q),
        time=state.time,
        step=state.step,
        N=args.grid_size,
        days=args.days
    )
    print(f"\n✓ Final state saved to {output_dir / 'final_state.npz'}")


if __name__ == "__main__":
    main()

