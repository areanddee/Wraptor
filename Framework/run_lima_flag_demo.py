"""
Lima Flag Demonstration - Run diffusion and create movie

This script:
1. Runs thermal diffusion with Lima Flag initial condition
2. Saves output in Zarr format
3. Generates geometry file
4. Creates a movie using existing cubesphere_diffusion_viz.py

Usage:
    python run_lima_flag_demo.py --grid-size 120 --days 10 --save-freq 12
    
    --save-freq is now in HOURS (not steps)
"""

import os
import sys
import argparse
import numpy as np
import zarr
from pathlib import Path

# Add Solvers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Solvers'))

# Import our diffusion solver
from fv_cubesphere_diffusion import CubedSphereDiffusion, equiangular_to_xyz_face

def create_geometry_file(N, output_path):
    """
    Create geometry Zarr file for visualization.
    
    Args:
        N: Grid resolution
        output_path: Path to save geometry (e.g., 'ICs/data/geometry/cubesphere_ne30.zarr')
    """
    print(f"\nCreating geometry file...")
    print(f"  Path: {output_path}")
    
    # Create grid
    xi1_1d = np.linspace(-np.pi/4, np.pi/4, N)
    xi2_1d = np.linspace(-np.pi/4, np.pi/4, N)
    XI1, XI2 = np.meshgrid(xi1_1d, xi2_1d, indexing='ij')
    
    # Compute cell centers for all faces
    X_all = np.zeros((6, N, N))
    Y_all = np.zeros((6, N, N))
    Z_all = np.zeros((6, N, N))
    
    for face in range(6):
        X, Y, Z = equiangular_to_xyz_face(XI1, XI2, face)
        X_all[face] = np.array(X)
        Y_all[face] = np.array(Y)
        Z_all[face] = np.array(Z)
    
    # Save to Zarr
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    store = zarr.open(str(output_path), mode='w')
    store.create_dataset('grid/X', data=X_all, chunks=(1, N, N))
    store.create_dataset('grid/Y', data=Y_all, chunks=(1, N, N))
    store.create_dataset('grid/Z', data=Z_all, chunks=(1, N, N))
    
    print(f"  ✓ Geometry saved: {X_all.shape}")


def run_diffusion_with_output(N=30, days=30.0, kappa=5e5, save_freq_hours=12, 
                              config_file=None, output_dir='lima_flag_output'):
    """
    Run diffusion and save output in Zarr format.
    
    Args:
        N: Grid resolution
        days: Simulation length [days]
        kappa: Diffusion coefficient [m²/s]
        save_freq_hours: Save frequency in hours
        config_file: Config file path
        output_dir: Output directory name
        
    Returns:
        output_path: Path to output directory
    """
    print("="*70)
    print("LIMA FLAG DEMONSTRATION - THERMAL DIFFUSION")
    print("="*70)
    print(f"Resolution: {N}×{N} per face")
    print(f"Duration: {days} days")
    print(f"Diffusivity: κ = {kappa:.2e} m²/s")
    print(f"Output directory: {output_dir}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / 'output').mkdir(exist_ok=True)
    
    # Set default config file if none provided
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), '..', 'Config', 
                                    'config_diffusion.yaml')
    
    # Create solver
    solver = CubedSphereDiffusion(N=N, kappa=kappa, config_file=config_file)
    
    # Initialize
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    # Time integration setup
    dt = 1800.0  # 30 minutes
    total_time = days * 86400.0
    n_steps = int(total_time / dt)
    
    # Convert save frequency from hours to steps
    save_freq_seconds = save_freq_hours * 3600.0
    save_freq = int(save_freq_seconds / dt)
    if save_freq < 1:
        save_freq = 1
        print(f"⚠ Warning: save_freq_hours too small, setting to 1 step")
    
    # Check CFL stability
    dx_physical = solver.dx * 6.371e6  # Convert to meters
    dt_cfl_max = 0.25 * dx_physical**2 / kappa
    cfl_number = dt / dt_cfl_max
    
    print(f"\nTime integration:")
    print(f"  dt: {dt:.1f} s ({dt/60:.1f} min)")
    print(f"  CFL: {cfl_number:.3f} {'(STABLE)' if cfl_number <= 0.5 else '(WARNING)'}")
    print(f"  n_steps: {n_steps}")
    print(f"  save_freq: every {save_freq_hours} hours ({save_freq} steps)")
    
    # Storage for Zarr output
    n_saves = n_steps // save_freq + 1
    T_history = []
    times_history = []
    
    # Initial frame
    T_history.append(np.array(state.T))
    times_history.append(0.0)
    
    # Compile step function
    print(f"\nJIT compiling step function...")
    state = solver.step(state, dt)
    print(f"  ✓ Compilation complete")
    
    # Time integration
    print(f"\nRunning {n_steps} steps...")
    
    initial_heat = solver.get_diagnostics(state)['heat_content']
    
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        # Save periodically
        if (step + 1) % save_freq == 0:
            T_history.append(np.array(state.T))
            times_history.append(state.time)
            
            diag = solver.get_diagnostics(state)
            heat_error = (diag['heat_content'] - initial_heat) / initial_heat
            
            print(f"  Step {step+1:4d} (day {diag['time']/86400:5.2f}): "
                  f"T_max={diag['T_max']:6.1f}K, "
                  f"Face0={diag['T_face0_max']:6.1f}K, "
                  f"Eq={diag['T_equatorial_max']:6.1f}K, "
                  f"Heat_err={heat_error:.2e}")
    
    # Save to Zarr
    print(f"\nSaving output to Zarr...")
    zarr_path = output_path / 'output' / 'output.zarr'
    store = zarr.open(str(zarr_path), mode='w')
    
    # Temperature array: (n_frames, 6, N, N)
    T_array = np.array(T_history)
    store.create_dataset('T', data=T_array, chunks=(1, 6, N, N))
    
    # Metadata: (n_frames, 2) - times and steps
    metadata = np.column_stack([
        np.array(times_history),
        np.arange(len(times_history)) * save_freq
    ])
    store.create_dataset('metadata', data=metadata, chunks=(len(times_history), 2))
    
    print(f"  ✓ Saved {len(T_history)} frames to {zarr_path}")
    
    # Final diagnostics
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    final_diag = solver.get_diagnostics(state)
    heat_error_final = (final_diag['heat_content'] - initial_heat) / initial_heat
    
    print(f"Heat conservation:")
    print(f"  Initial: {initial_heat:.6e}")
    print(f"  Final:   {final_diag['heat_content']:.6e}")
    print(f"  Error:   {heat_error_final:.2e}")
    print(f"\nOutput saved to: {output_path}")
    print(f"  - Temperature data: output/output.zarr")
    print(f"  - {len(T_history)} frames over {days} days")
    print(f"{'='*70}\n")
    
    return output_path


def generate_movie(output_dir, N, fps=15, dpi=150, elev=30, azim=45):
    """
    Generate movie using existing visualization tool.
    
    Args:
        output_dir: Directory containing output/output.zarr
        N: Grid resolution
        fps: Frames per second
        dpi: Resolution
        elev: Elevation angle
        azim: Azimuth angle
    """
    print(f"\n{'='*70}")
    print("GENERATING MOVIE")
    print(f"{'='*70}")
    
    # Path to visualization script
    viz_script = Path(__file__).parent.parent / 'Analysis' / 'cubesphere_diffusion_viz.py'
    
    if not viz_script.exists():
        print(f"⚠ Visualization script not found: {viz_script}")
        print(f"  You can manually create a movie later using:")
        print(f"  cd {output_dir}")
        print(f"  python ../../Analysis/cubesphere_diffusion_viz.py --movie --fps {fps}")
        return
    
    # Create geometry file if it doesn't exist
    geometry_dir = Path(__file__).parent.parent / 'ICs' / 'data' / 'geometry'
    geometry_path = geometry_dir / f'cubesphere_ne{N}.zarr'
    
    if not geometry_path.exists():
        create_geometry_file(N, geometry_path)
    else:
        print(f"  ✓ Geometry file exists: {geometry_path}")
    
    # Run visualization script
    output_movie = Path(output_dir) / 'lima_flag_movie.mp4'
    
    cmd = (
        f'cd {output_dir} && '
        f'python {viz_script} '
        f'--movie '
        f'--fps {fps} '
        f'--dpi {dpi} '
        f'--elev {elev} '
        f'--azim {azim} '
        f'--cmap inferno '
        f'--output lima_flag_movie.mp4'
    )
    
    print(f"  Running: {cmd}")
    print()
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print(f"\n{'='*70}")
        print(f"✓ MOVIE GENERATED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"  Location: {output_movie}")
        print(f"  Duration: ~{len(list(Path(output_dir).glob('*.png'))) / fps:.1f} seconds @ {fps} fps")
        print(f"{'='*70}\n")
    else:
        print(f"\n⚠ Movie generation failed (exit code: {exit_code})")
        print(f"  You can try manually:")
        print(f"  cd {output_dir}")
        print(f"  python ../../Analysis/cubesphere_diffusion_viz.py --movie")


def main():
    parser = argparse.ArgumentParser(
        description='Lima Flag Demonstration - Thermal Diffusion on Cubed-Sphere',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-size', type=int, default=30,
                        help='Grid resolution (N×N cells per face)')
    parser.add_argument('--days', type=float, default=30.0,
                        help='Simulation length in days')
    parser.add_argument('--kappa', type=float, default=5e5,
                        help='Diffusion coefficient [m²/s]')
    parser.add_argument('--save-freq', type=int, default=12,
                        help='Save frequency in hours (default: 12 hours = twice per day)')
    parser.add_argument('--config', type=str,
                        default=None,
                        help='Path to config file (default: ../Config/config_diffusion.yaml)')
    parser.add_argument('--output-dir', type=str, default='lima_flag_output',
                        help='Output directory name')
    parser.add_argument('--skip-movie', action='store_true',
                        help='Skip movie generation (data only)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Movie frames per second')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Movie resolution (DPI)')
    parser.add_argument('--elev', type=float, default=30,
                        help='Viewing elevation angle')
    parser.add_argument('--azim', type=float, default=45,
                        help='Viewing azimuth angle')
    
    args = parser.parse_args()
    
    # Run diffusion simulation
    output_path = run_diffusion_with_output(
        N=args.grid_size,
        days=args.days,
        kappa=args.kappa,
        save_freq_hours=args.save_freq,
        config_file=args.config,
        output_dir=args.output_dir
    )
    
    # Generate movie (unless skipped)
    if not args.skip_movie:
        generate_movie(
            output_dir=output_path,
            N=args.grid_size,
            fps=args.fps,
            dpi=args.dpi,
            elev=args.elev,
            azim=args.azim
        )
    else:
        print("✓ Simulation complete! (Movie generation skipped)")
        print(f"  To generate movie later:")
        print(f"  cd {output_path}")
        print(f"  python ../../Analysis/cubesphere_diffusion_viz.py --movie")


if __name__ == "__main__":
    main()

