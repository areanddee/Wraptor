#!/usr/bin/env python3
"""
Spherical Visualization Tool for Wraptor Solvers

Self-contained visualization that uses solver geometry directly.
No external geometry files needed!

Supports:
- Diffusion (temperature) and Advection (tracer) visualization
- Single frames, multiple frames, and movies
- Interactive 3D rotation
- Progress bars
- Linear and logarithmic color scales

Usage:
    # Quick test: run solver and visualize
    python Analysis/visualize_sphere.py --solver diffusion --days 5 --save-freq 1
    
    # Visualize from existing Zarr output
    python Analysis/visualize_sphere.py --zarr output/output.zarr
    
    # Generate movie from frames directory
    python Analysis/visualize_sphere.py --frames-dir output/frames --movie

Author: Wraptor Development Team
Date: December 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for stability
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_solver_coordinates(solver_type: str, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get Cartesian coordinates from solver's own geometry module.
    
    Returns:
        X, Y, Z: (6, N, N) coordinate arrays on unit sphere
    """
    from Solvers.geometry import CubedSphereGeometry
    
    geometry = CubedSphereGeometry.create(N)
    
    # Get coordinates for all faces using the correct method
    X, Y, Z = geometry.get_xyz_all_faces()
    
    return X, Y, Z


def interpolate_to_latlon(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                          field: np.ndarray, 
                          nlat: int = 180, nlon: int = 360,
                          method: str = 'linear') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate cubed-sphere data to regular lat-lon grid.
    
    Args:
        X, Y, Z: (6, N, N) Cartesian coordinates
        field: (6, N, N) field values
        nlat, nlon: interpolation resolution
        method: 'linear' or 'cubic'
        
    Returns:
        X_sphere, Y_sphere, Z_sphere, field_interp: (nlat, nlon) arrays
    """
    # Flatten
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    field_flat = field.flatten()
    
    # Convert to lat-lon
    lat_flat = np.arcsin(np.clip(z_flat, -1, 1))
    lon_flat = np.arctan2(y_flat, x_flat)
    
    # Create regular lat-lon grid
    lat_grid = np.linspace(-np.pi/2, np.pi/2, nlat)
    lon_grid = np.linspace(-np.pi, np.pi, nlon)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolate
    points = np.column_stack([lon_flat, lat_flat])
    field_interp = griddata(points, field_flat, (LON, LAT), method=method)
    
    # Fill NaN at poles
    if np.any(np.isnan(field_interp)):
        mask = np.isnan(field_interp)
        field_interp[mask] = griddata(points, field_flat, (LON[mask], LAT[mask]), method='nearest')
    
    # Convert back to Cartesian
    X_sphere = np.cos(LAT) * np.cos(LON)
    Y_sphere = np.cos(LAT) * np.sin(LON)
    Z_sphere = np.sin(LAT)
    
    return X_sphere, Y_sphere, Z_sphere, field_interp


def plot_sphere(field: np.ndarray, 
                X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                title: str = "Field",
                cmap: str = 'inferno',
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                log_scale: bool = False,
                elev: float = 30,
                azim: float = 45,
                nlat: int = 180,
                nlon: int = 360,
                output_path: Optional[str] = None,
                dpi: int = 150,
                figsize: Tuple[float, float] = (10, 8),  # Will be adjusted for even pixel dims
                colorbar_label: str = "Value") -> plt.Figure:
    """
    Plot a field on the sphere.
    
    Args:
        field: (6, N, N) field values
        X, Y, Z: (6, N, N) Cartesian coordinates
        title: plot title
        cmap: colormap name
        vmin, vmax: color scale limits
        log_scale: use logarithmic color scale
        elev, azim: viewing angles
        nlat, nlon: interpolation resolution
        output_path: save path (None = return figure)
        dpi: output resolution
        figsize: figure size
        colorbar_label: label for colorbar
        
    Returns:
        matplotlib Figure
    """
    # Interpolate
    X_sphere, Y_sphere, Z_sphere, field_interp = interpolate_to_latlon(
        X, Y, Z, field, nlat=nlat, nlon=nlon
    )
    
    # Set color limits
    if vmin is None:
        vmin = float(np.nanmin(field))
    if vmax is None:
        vmax = float(np.nanmax(field))
    
    if log_scale:
        vmin = max(vmin, 1e-10)  # Avoid log(0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        colors_normalized = norm(np.clip(field_interp, vmin, vmax))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        if vmax > vmin:
            colors_normalized = (field_interp - vmin) / (vmax - vmin)
        else:
            colors_normalized = np.zeros_like(field_interp)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(
        X_sphere, Y_sphere, Z_sphere,
        facecolors=plt.colormaps[cmap](colors_normalized),
        rstride=1, cstride=1,
        shade=False,
        antialiased=True
    )
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label(colorbar_label, fontsize=11)
    
    # Setup axes
    limit = 1.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=13, pad=15)
    ax.view_init(elev=elev, azim=azim)
    
    # Clean up panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


def generate_frames(fields: np.ndarray,
                   times: np.ndarray,
                   X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                   output_dir: str,
                   field_name: str = "Temperature",
                   cmap: str = 'inferno',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   log_scale: bool = False,
                   elev: float = 30,
                   azim: float = 45,
                   nlat: int = 180,
                   nlon: int = 360,
                   dpi: int = 150,
                   skip_first: bool = False,
                   progress: bool = True) -> int:
    """
    Generate PNG frames from field time series.
    
    Args:
        fields: (n_frames, 6, N, N) field time series
        times: (n_frames,) times in seconds
        X, Y, Z: (6, N, N) Cartesian coordinates
        output_dir: directory to save frames
        field_name: name for title
        cmap: colormap
        vmin, vmax: color limits (None = auto from all frames)
        log_scale: logarithmic color scale
        elev, azim: viewing angles
        nlat, nlon: interpolation resolution
        dpi: frame resolution
        skip_first: skip frame 0
        progress: show progress bar
        
    Returns:
        Number of frames generated
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_frames = len(fields)
    start_frame = 1 if skip_first else 0
    
    # Auto color limits
    if vmin is None:
        vmin = float(np.nanmin(fields))
    if vmax is None:
        vmax = float(np.nanmax(fields))
    
    print(f"\nGenerating {n_frames - start_frame} frames...")
    print(f"  Color range: [{vmin:.2f}, {vmax:.2f}]")
    print(f"  Output: {output_dir}/")
    
    for i in range(start_frame, n_frames):
        if progress:
            pct = 100 * (i - start_frame + 1) / (n_frames - start_frame)
            bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:5.1f}%", end='', flush=True)
        
        time_days = times[i] / 86400.0
        title = f"{field_name} - Day {time_days:.2f}"
        
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        
        plot_sphere(
            fields[i], X, Y, Z,
            title=title,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            log_scale=log_scale,
            elev=elev, azim=azim,
            nlat=nlat, nlon=nlon,
            output_path=output_path,
            dpi=dpi,
            colorbar_label=field_name
        )
    
    if progress:
        print()  # Newline after progress bar
    
    print(f"  ✓ Generated {n_frames - start_frame} frames")
    return n_frames - start_frame


def frames_to_movie(frames_dir: str,
                   output_path: str = 'movie.mp4',
                   fps: int = 15,
                   pattern: str = 'frame_*.png') -> bool:
    """
    Convert PNG frames to MP4 movie using ffmpeg.
    
    Args:
        frames_dir: directory containing frame PNGs
        output_path: output movie path
        fps: frames per second
        pattern: frame filename pattern
        
    Returns:
        True if successful
    """
    import subprocess
    import glob
    
    # Check frames exist
    frames = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not frames:
        print(f"Error: No frames found matching {pattern} in {frames_dir}")
        return False
    
    print(f"\nGenerating movie from {len(frames)} frames...")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames)/fps:.1f} seconds")
    print(f"  Output: {output_path}")
    
    # Build ffmpeg command
    # Use frame_%04d.png pattern for sequential frames
    input_pattern = os.path.join(frames_dir, 'frame_%04d.png')
    
    # libx264 requires even dimensions, so we scale to nearest even size
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', input_pattern,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Round to even dimensions
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',  # Quality (lower = better, 18-28 is good range)
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Movie saved: {output_path}")
            return True
        else:
            print(f"Error: ffmpeg failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install with:")
        print("  brew install ffmpeg  # macOS")
        print("  sudo apt install ffmpeg  # Linux")
        return False


def run_and_visualize(solver_type: str,
                     N: int = 60,
                     days: float = 10,
                     save_freq_hours: float = 6,
                     output_dir: str = 'output',
                     make_movie: bool = True,
                     fps: int = 15,
                     **viz_kwargs) -> str:
    """
    Run a solver and generate visualization.
    
    Args:
        solver_type: 'diffusion' or 'advection'
        N: grid resolution
        days: simulation duration
        save_freq_hours: output frequency in hours
        output_dir: output directory
        make_movie: generate movie from frames
        fps: movie FPS
        **viz_kwargs: passed to generate_frames
        
    Returns:
        Path to movie or frames directory
    """
    import jax.numpy as jnp
    
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    
    print(f"\n{'='*60}")
    print(f"RUN AND VISUALIZE: {solver_type.upper()}")
    print(f"{'='*60}")
    print(f"  Grid: {N}×{N} per face")
    print(f"  Duration: {days} days")
    print(f"  Save frequency: {save_freq_hours} hours")
    
    # Get coordinates
    X, Y, Z = get_solver_coordinates(solver_type, N)
    
    # Load solver
    config_path = Path(__file__).parent.parent / 'Config'
    
    if solver_type == 'diffusion':
        from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
        config = config_path / 'config_diffusion_no_sharding.yaml'
        kappa = 5e5  # diffusivity
        solver = CubedSphereDiffusion(N=N, kappa=kappa, config_file=str(config))
        state = solver.initialize(pattern='quadrant', T_hot=600.0)
        # Compute CFL timestep: dt_max = 0.25 * dx^2 / kappa
        dx_physical = solver.dx * solver.planet.R_sphere
        dt_max = 0.25 * dx_physical**2 / kappa
        dt = 0.5 * dt_max
        field_name = 'Temperature (K)'
        get_field = lambda s: np.array(s.T)
        cmap = 'inferno'
        log_scale = False
    else:
        from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
        config = config_path / 'config_plr_advection_no_sharding.yaml'
        solver = PLRCubeSphereAdvection(N=N, config_file=str(config))
        state = solver.initialize()
        V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
        V_max = float(jnp.max(V_mag))
        dt = 0.5 * (solver.dx * solver.planet.R_sphere) / V_max
        field_name = 'Tracer'
        get_field = lambda s: np.array(s.q)
        cmap = 'viridis'
        log_scale = False
    
    # Calculate steps
    save_freq_seconds = save_freq_hours * 3600
    steps_per_save = max(1, int(save_freq_seconds / dt))
    total_steps = int(days * 86400 / dt)
    n_saves = total_steps // steps_per_save + 1
    
    print(f"  dt: {dt:.1f} s ({dt/60:.1f} min)")
    print(f"  Steps: {total_steps}")
    print(f"  Frames: {n_saves}")
    
    # Run and collect frames
    fields = [get_field(state)]
    times = [0.0]
    
    print(f"\nRunning simulation...")
    
    for step in range(1, total_steps + 1):
        state = solver.step(state, dt)
        
        if step % steps_per_save == 0:
            fields.append(get_field(state))
            times.append(state.time)
            
            pct = 100 * step / total_steps
            bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:5.1f}% (Day {state.time/86400:.1f})", end='', flush=True)
    
    print()  # Newline
    
    fields = np.array(fields)
    times = np.array(times)
    
    print(f"  ✓ Simulation complete: {len(fields)} frames")
    
    # Generate frames (merge viz_kwargs but don't override our defaults)
    frame_kwargs = {
        'cmap': cmap,
        'log_scale': log_scale,
        'skip_first': True,  # Skip t=0 which often looks bad
    }
    # Allow viz_kwargs to override if explicitly provided
    for k, v in viz_kwargs.items():
        if v is not None:  # Only override if explicitly set
            frame_kwargs[k] = v
    
    generate_frames(
        fields, times, X, Y, Z,
        output_dir=frames_dir,
        field_name=field_name,
        **frame_kwargs
    )
    
    # Make movie
    if make_movie:
        movie_path = os.path.join(output_dir, f'{solver_type}_movie.mp4')
        frames_to_movie(frames_dir, movie_path, fps=fps)
        return movie_path
    
    return frames_dir


def main():
    parser = argparse.ArgumentParser(
        description='Spherical Visualization for Wraptor Solvers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run diffusion and make movie
    python Analysis/visualize_sphere.py --solver diffusion --days 10 --save-freq 6
    
    # Run advection at higher resolution
    python Analysis/visualize_sphere.py --solver advection --N 120 --days 12 --fps 30
    
    # Generate movie from existing frames
    python Analysis/visualize_sphere.py --frames-dir output/frames --movie --fps 20
        """
    )
    
    # Mode selection
    parser.add_argument('--solver', type=str, choices=['diffusion', 'advection'],
                       help='Run solver and visualize')
    parser.add_argument('--frames-dir', type=str,
                       help='Generate movie from existing frames directory')
    
    # Solver options
    parser.add_argument('--N', type=int, default=60, help='Grid resolution (default: 60)')
    parser.add_argument('--days', type=float, default=10, help='Simulation days (default: 10)')
    parser.add_argument('--save-freq', type=float, default=6, 
                       help='Save frequency in hours (default: 6)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--movie', action='store_true', help='Generate movie')
    parser.add_argument('--fps', type=int, default=15, help='Movie FPS (default: 15)')
    
    # Visualization options
    parser.add_argument('--cmap', type=str, default='inferno', help='Colormap')
    parser.add_argument('--vmin', type=float, help='Color scale minimum')
    parser.add_argument('--vmax', type=float, help='Color scale maximum')
    parser.add_argument('--log-scale', action='store_true', help='Logarithmic color scale')
    parser.add_argument('--elev', type=float, default=30, help='Elevation angle (default: 30)')
    parser.add_argument('--azim', type=float, default=45, help='Azimuth angle (default: 45)')
    parser.add_argument('--nlat', type=int, default=180, help='Lat interpolation (default: 180)')
    parser.add_argument('--nlon', type=int, default=360, help='Lon interpolation (default: 360)')
    parser.add_argument('--dpi', type=int, default=150, help='Frame DPI (default: 150)')
    
    args = parser.parse_args()
    
    if args.solver:
        # Run solver and visualize
        run_and_visualize(
            solver_type=args.solver,
            N=args.N,
            days=args.days,
            save_freq_hours=args.save_freq,
            output_dir=args.output_dir,
            make_movie=True,
            fps=args.fps,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            log_scale=args.log_scale,
            elev=args.elev,
            azim=args.azim,
            nlat=args.nlat,
            nlon=args.nlon,
            dpi=args.dpi
        )
    
    elif args.frames_dir:
        # Generate movie from frames
        if args.movie:
            movie_path = os.path.join(args.output_dir, 'movie.mp4')
            frames_to_movie(args.frames_dir, movie_path, fps=args.fps)
        else:
            print("Use --movie flag to generate movie from frames")
    
    else:
        parser.print_help()
        print("\nError: Specify --solver or --frames-dir")
        sys.exit(1)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()

