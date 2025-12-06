"""
Cosine Bell Advection Demo - PLR 2nd-order Advection

This script:
1. Runs PLR advection with solid-body rotation (Williamson test case 1)
2. Saves output in Zarr format for analysis
3. Creates work directory for isolated runs

Usage:
    python run_cosine_bell_demo.py --grid-size 120 --days 12 --save-freq 12
"""

import os
import sys
import argparse
import numpy as np
import zarr
from pathlib import Path

# Add Solvers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Solvers'))

from fv_plr_cubesphere_adv import PLRCubeSphereAdvection

def run_advection_with_output(N=60, days=12.0, save_freq_hours=12,
                               config_file=None, output_dir='cosine_bell_output'):
    """
    Run PLR advection and save output in Zarr format.
    
    Args:
        N: Grid resolution
        days: Simulation length [days]
        save_freq_hours: Save frequency in hours
        config_file: Config file path
        output_dir: Output directory name
        
    Returns:
        output_path: Path to output directory
    """
    print("="*70)
    print("COSINE BELL ADVECTION DEMO - PLR 2ND-ORDER")
    print("="*70)
    print(f"Resolution: {N}×{N} per face")
    print(f"Duration: {days} days")
    print(f"Output directory: {output_dir}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / 'output').mkdir(exist_ok=True)
    
    # Create solver
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), '..', 'Config', 
                                    'config_plr_advection.yaml')
    
    solver = PLRCubeSphereAdvection(N=N, config_file=config_file)
    
    # Initialize with cosine bell
    state = solver.initialize()
    
    # Compute timestep from velocity field
    import jax.numpy as jnp
    R_SPHERE = 6.371e6
    V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
    V_max = float(jnp.max(V_mag))
    target_cfl = 0.5
    dt = target_cfl * (solver.dx * R_SPHERE) / V_max
    total_time = days * 86400.0
    n_steps = int(total_time / dt)
    
    # Convert save frequency from hours to steps
    save_freq_seconds = save_freq_hours * 3600.0
    save_freq = int(save_freq_seconds / dt)
    if save_freq < 1:
        save_freq = 1
    
    print(f"\nTime integration:")
    print(f"  CFL: {target_cfl}")
    print(f"  V_max: {V_max:.2f} m/s")
    print(f"  dt: {dt:.1f} s ({dt/60:.1f} min)")
    print(f"  n_steps: {n_steps}")
    print(f"  save_freq: every {save_freq_hours} hours ({save_freq} steps)")
    
    # Storage for Zarr output
    q_history = []
    times_history = []
    
    # Initial frame
    q_history.append(np.array(state.q))
    times_history.append(0.0)
    
    # Compile step function
    print(f"\nJIT compiling step function...")
    state = solver.step(state, dt)
    print(f"  ✓ Compilation complete")
    
    # Time integration
    print(f"\nRunning {n_steps} steps...")
    
    initial_mass = np.sum(state.q * solver.sqrtG_all)
    
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        # Save periodically
        if (step + 1) % save_freq == 0:
            q_history.append(np.array(state.q))
            times_history.append(state.time)
            
            # Diagnostics
            mass = np.sum(state.q * solver.sqrtG_all)
            mass_error = (mass - initial_mass) / initial_mass
            q_max = np.max(state.q)
            
            print(f"  Step {step+1:4d} (day {state.time/86400:5.2f}): "
                  f"q_max={q_max:6.1f}, "
                  f"mass_err={mass_error:.2e}")
    
    # Save to Zarr
    print(f"\nSaving output to Zarr...")
    zarr_path = output_path / 'output' / 'output.zarr'
    store = zarr.open(str(zarr_path), mode='w')
    
    # Concentration array: (n_frames, 6, N, N)
    q_array = np.array(q_history)
    store.create_dataset('q', data=q_array, chunks=(1, 6, N, N))
    
    # Metadata: (n_frames, 2) - times and steps
    metadata = np.column_stack([
        np.array(times_history),
        np.arange(len(times_history)) * save_freq
    ])
    store.create_dataset('metadata', data=metadata, chunks=(len(times_history), 2))
    
    # Times in days (for convenience)
    times_days = np.array(times_history) / 86400.0
    store.create_dataset('times_days', data=times_days)
    
    print(f"  ✓ Saved {len(q_history)} frames to {zarr_path}")
    
    # Final diagnostics
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    final_mass = np.sum(state.q * solver.sqrtG_all)
    mass_error_final = (final_mass - initial_mass) / initial_mass
    
    print(f"Mass conservation:")
    print(f"  Initial: {initial_mass:.6e}")
    print(f"  Final:   {final_mass:.6e}")
    print(f"  Error:   {mass_error_final:.2e}")
    print(f"\nConcentration:")
    print(f"  Initial max: {np.max(q_history[0]):.1f}")
    print(f"  Final max:   {np.max(state.q):.1f}")
    print(f"\nOutput saved to: {output_path}")
    print(f"  - Concentration data: output/output.zarr")
    print(f"  - {len(q_history)} frames over {days} days")
    print(f"{'='*70}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Cosine Bell Advection Demo - PLR 2nd-order',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-size', type=int, default=60,
                        help='Grid resolution (N×N cells per face)')
    parser.add_argument('--days', type=float, default=12.0,
                        help='Simulation length in days')
    parser.add_argument('--save-freq', type=int, default=12,
                        help='Save frequency in hours (default: 12 hours = twice per day)')
    parser.add_argument('--config', type=str,
                        default='../Config/config_plr_advection.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='cosine_bell_output',
                        help='Output directory name')
    
    args = parser.parse_args()
    
    # Run advection simulation
    output_path = run_advection_with_output(
        N=args.grid_size,
        days=args.days,
        save_freq_hours=args.save_freq,
        config_file=args.config,
        output_dir=args.output_dir
    )
    
    print("✓ Simulation complete!")
    print(f"\nTo visualize (placeholder - need to adapt movie script):")
    print(f"  cd ../Analysis")
    print(f"  # TODO: Create cosine_bell visualization script")


if __name__ == "__main__":
    main()

